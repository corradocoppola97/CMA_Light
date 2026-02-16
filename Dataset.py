import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os


class Dataset:
    """
    Class to define the dasat and its properties
    """

    def __init__(self,
                 csvFile: str,
                 scaling: bool = False,
                 perc_test: float = 0.25,
                 frac: float = 1.0
                 ):
        """
        :param csvFile: path to the dataset to read
        :param scaling: if True/1 data are scaled
        :param perc_test: percentage of data used for testing
        """

        self.dataset = pd.read_csv(csvFile)
        self.csv = csvFile.split('/')[-1]
        self.dataset = self.dataset.sample(frac=frac, random_state=0)  # così ho fatto anche reshuffling

        if scaling == 1:
            scaler = MinMaxScaler()
            self.dataset = scaler.fit_transform(self.dataset)

        self.dataset = torch.Tensor(self.dataset).float()

        # extract train and test
        self.x = self.dataset[:, :-1]
        self.y = self.dataset[:, -1]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=perc_test, random_state=0)
        self.P = self.x_train.shape[0]
        self.P_test = self.x_test.shape[0]
        self.n = self.x_train.shape[1]
        self.idx = None


    def minibatch(self, first: int, last: int, test = False):
        """Extract a minibatch of observations from the starting dataset"""
        if test == False:
            if last > self.P:
                last = self.P
            self.x_train_mb = self.x_train[first:last, :]
            self.y_train_mb = self.y_train[first:last].flatten()
        else:
            self.x_test_mb = self.x_test[first:last, :]
            self.y_test_mb = self.y_test[first:last].flatten()

    def __str__(self):
        description = 'Dataset: {}.\nNumber of samples: {}. Number of variables: {}.'.format(self.csv, self.x.shape[0], self.x.shape[1])
        description = description + '\nTraining size: {}. Test size: {}.'.format(self.x_train.shape[1], self.x_test.shape[1])
        return description

    def get_idx(self,seed=100):
        '''
        Define the order in which the sample will be considered within the minibatches
        '''
        np.random.seed(seed)
        self.idx = np.arange(self.x_train.shape[0])
        np.random.shuffle(self.idx)
        return self.idx

    def reshuffle(self,seed=100):
        idx = self.get_idx(seed=seed)
        self.x_train = self.x_train[idx]
        self.y_train = self.y_train[idx]


def create_dataset(dataset: str, frac: float = 1.0):
    if dataset not in {'cifar10', 'cifar100'}:
        path = os.getcwd() +'/dataset/' + dataset+'.csv'
        csv_file_path = path
    else:
        csv_file_path = dataset
    try:
        dataset = Dataset(csv_file_path, scaling=True, frac=frac)
    except FileNotFoundError as e:
        print("data were not detected correctly. Check your cwd and the location of your dataset folder")
        print(f"cwd: {os.getcwd()}")
        raise e

    return dataset
