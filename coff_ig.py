import torch
import torchvision
import copy


def get_w(model: torch.nn.Module) -> torch.Tensor:
    """Flatten model parameters into a single vector."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_w(model: torch.nn.Module, w: torch.Tensor) -> None:
    """Load parameters from a flat vector into the model."""
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.copy_(w[idx:idx + numel].view_as(p))
            idx += numel


class Coff_Ig(torch.optim.Optimizer):

    def __init__(self, params, alpha=0.05, zeta=0.05, eps=1e-3, theta=0.5,
                 delta=0.9, gamma=1e-6, tau=1e-2, verbose=False, max_it_EDFL=100,
                 verbose_EDFL=False):

        defaults = dict(alpha=alpha, zeta=zeta, eps=eps, theta=theta,
                        delta=delta, gamma=gamma, verbose=verbose,
                        maximize=False, tau=tau, max_it_EDFL=max_it_EDFL,
                        verbose_EDFL=verbose_EDFL)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('verbose', False)
            group.setdefault('maximize', False)

    # setters for scalars
    def set_zeta(self, zeta: float): self.param_groups[0]['zeta'] = zeta
    def set_fw0(self, fw0: float): self.fw0 = fw0
    def set_f_tilde(self, f_tilde: float): self.f_tilde = f_tilde
    def set_phi(self, phi: float): self.phi = phi

    def step(self, closure=None, *args, **kwargs):
        """One optimization step."""
        loss = None
        with torch.no_grad():
            if closure is not None:
                loss = closure(*args, **kwargs)
            else:
                for group in self.param_groups:
                    zeta = group['zeta']
                    for p in group['params']:
                        if p.grad is not None:
                            p.add_(p.grad, alpha=-zeta)
        return loss

    def EDFL(self,
             mod: torchvision.models,
             dl_train: torch.utils.data.DataLoader,
             w_before: torch.Tensor,
             f_tilde: float,
             d_k: torch.Tensor,
             closure: callable,
             device: torch.device,
             criterion: torch.nn):

        zeta = self.param_groups[0]['zeta']
        gamma, delta = self.defaults['gamma'], self.defaults['delta']
        alpha, nfev = zeta, 0
        verbose = self.defaults['verbose_EDFL']


        sample_model = copy.deepcopy(mod)
        sample_model.load_state_dict(mod.state_dict())

        real_loss = closure(dl_train, device, sample_model, criterion)
        nfev += 1
        if verbose:
            print(f'Starting EDFL with zeta={zeta}, alpha={alpha}, f_tilde={f_tilde}, real_loss_before={real_loss}')

        if f_tilde > min(real_loss, self.fw0):
            if verbose: print('fail: ALPHA = 0')
            return 0, nfev, f_tilde

        # helper to set params quickly
        def apply_w(model, w_vec):
            set_w(model, w_vec)

        w_prova = w_before + d_k * (alpha / delta)
        apply_w(sample_model, w_prova)
        cur_loss = closure(dl_train, device, sample_model, criterion)
        nfev += 1

        idx, f_j = 0, f_tilde
        while cur_loss <= min(f_j, real_loss - gamma * alpha * torch.norm(d_k) ** 2) \
                and idx <= self.defaults['max_it_EDFL']:
            if verbose: print(f'idx={idx} cur_loss={cur_loss}')
            f_j, alpha = cur_loss, alpha / delta
            w_prova = w_before + d_k * (alpha / delta)
            apply_w(sample_model, w_prova)
            cur_loss = closure(dl_train, device, sample_model, criterion)
            nfev, idx = nfev + 1, idx + 1

        return alpha, nfev, f_j

    def control_step(self,
                     model: torchvision.models,
                     w_before: torch.Tensor,
                     closure: callable,
                     dl_train: torch.utils.data.DataLoader,
                     device: torch.device,
                     criterion: torch.nn,
                     history: dict,
                     epoch: int):

        zeta = self.param_groups[0]['zeta']
        gamma, theta, tau = self.defaults['gamma'], self.defaults['theta'], self.defaults['tau']
        verbose = self.param_groups[0]['verbose']

        f_tilde, fw0, phi = self.f_tilde, self.fw0, self.phi
        w_after = get_w(model)
        d = (w_after - w_before) / zeta  # Descent direction

        if f_tilde < min(fw0, phi - gamma * zeta):  # best case
            f_after = f_tilde
            history['accepted'].append(epoch)
            history['Exit'].append('7')
            if verbose: print('ok inner cycle')

        else:
            set_w(model, w_before)  # rollback
            if verbose: print('back to w_k')

            if torch.norm(d) <= tau * zeta:  # small step
                if verbose: print('||d|| small --> Step size reduced')
                self.set_zeta(zeta * theta)
                if f_tilde <= fw0:
                    alpha, f_after = zeta, f_tilde
                    set_w(model, w_before + alpha * d)
                    history['Exit'].append('10a')
                else:
                    alpha, f_after = 0, phi
                    history['Exit'].append('9b')

            else:  # try EDFL
                if verbose: print('Executing EDFL')
                alpha, nf_EDFL, f_after_LS = self.EDFL(model, dl_train, w_before,
                                                       f_tilde, d, closure,
                                                       device, criterion)
                history['nfev'] += nf_EDFL
                if alpha * torch.norm(d) ** 2 <= tau * zeta:  # reduce step
                    self.set_zeta(zeta * theta)
                    if alpha > 0:
                        if verbose: print('LS accepted')
                        f_after = f_after_LS
                        history['Exit'].append('15a')
                    elif f_tilde <= fw0:
                        if verbose: print('Step reduced but w_tilde accepted')
                        alpha, f_after = zeta, f_tilde
                        history['Exit'].append('15b')
                    else:
                        if verbose: print('Total fail')
                        alpha, f_after = 0, phi
                        history['Exit'].append('15c')
                else:  # total success
                    f_after = f_after_LS
                    self.set_zeta(alpha)
                    history['Exit'].append('16')

            if verbose: print(f'Final alpha={alpha}, Current step-size zeta={zeta}')
            if alpha > 0:  # update model
                set_w(model, w_before + alpha * d)

        if verbose:
            print(f'phi_before={phi:.3e} f_tilde={f_tilde:.3e} f_after={f_after:.3e} Exit={history["Exit"][-1]}')
            print(f'Step-size: {self.param_groups[0]["zeta"]:.3e}')

        history['step_size'].append(self.param_groups[0]['zeta'])
        return model, history, f_after, history['Exit'][-1]
