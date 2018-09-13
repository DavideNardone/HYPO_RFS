from __future__ import division
import hdf5storage
import numpy as np

np.set_printoptions(threshold=np.inf)
import scipy.io as sio


class Loader:

    def __init__(self, file_path, name, variables, format, k_fold=None):

        '''
        This Class provides several method for loading many type of dataset (matlab, csv, txt, etc)

        '''
        if format == 'matlab':  # classic workspace

            mc = sio.loadmat(file_path)

            for variable in variables:
                setattr(self, variable, mc[variable])

        elif format == 'matlab_struct':  # struct one level
            print ('Loading data...')

            mc = sio.loadmat(file_path)
            mc = mc[name][0, 0]

            for variable in variables:
                setattr(self, variable, mc[variable])

        elif format == 'custom_matlab':
            print ('Loading data...')

            mc = sio.loadmat(file_path)
            mc = mc[name][0, 0]

            for variable in variables:
                setattr(self, variable, mc[variable][0, 0])
        elif format == 'custom_matlab_2':
            print ('Loading data...')

            mc = sio.loadmat(file_path)
            mc = mc[name][0, 0]

            X_train = []
            y_train = []
            X_test = []
            y_test = []
            for variable in variables:
                kf_ind = []
                X_train = mc[variable]['X_train'][0, 0]
                X_test = mc[variable]['X_test'][0, 0]
                y_train = mc[variable]['y_train'][0, 0]
                y_test = mc[variable]['y_test'][0, 0]
                for i in xrange(0, k_fold):
                    kf_ind.append(mc[variable]['k_fold'][0, 0][0, i][0, 0])
                setattr(self, variable, kf_ind)
                setattr(self, 'X_train', X_train)
                setattr(self, 'X_test', X_test)
                setattr(self, 'y_train', y_train)
                setattr(self, 'y_test', y_test)


        elif format == 'matlab_v73':
            mc = hdf5storage.loadmat(file_path)

            for variable in variables:
                setattr(self, variable, mc[variable])

    def getVariables(self, variables):

        D = {}

        for variable in variables:
            D[variable] = getattr(self, variable)

        return D
    
class Dataset:

def __init__(self, X, y):

    self.data = X
    self.target = y.flatten()

    # removing any row with at least one NaN value
    # TODO: remove also the corresponding target value
    self.data = self.data[~np.isnan(self.data).any(axis=1)]

    self.num_sample, self.num_features = self.data.shape[0], self.data.shape[1]

    # retrieving unique label for Dataset
    self.classes = np.unique(self.target)

def standardizeDataset(self):

    # it simply standardize the data [mean 0 and std 1]

    if np.sum(np.std(self.data, axis=0)).astype('int32') == self.num_features and np.sum(
            np.mean(self.data, axis=0)) < 1 ** -7:
        print ('\tThe data were already standardized!')
    else:
        print ('Standardizing data....')
        self.data = StandardScaler().fit_transform(self.data)

    # it simply standardize the data [mean 0 and std 1]
    # self.data = preprocessing.scale(self.data)

def normalizeDataset(self, norm):
    # print ('Normalizing data....')

    normalizer = preprocessing.Normalizer(norm=norm)
    self.data = normalizer.fit_transform(self.data)

def scalingDataset(self):
    # print ('Scaling data between [0,1]...')

    min_max_scaler = preprocessing.MinMaxScaler()
    self.data = min_max_scaler.fit_transform(self.data)

def shufflingDataset(self):
    # print ('Shuffling data....')

    idx = np.random.permutation(self.data.shape[0])
    self.data = self.data[idx]
    self.target = self.target[idx]

# TODO
def mergeDataset(self, X1, X2):
    print ''

def split(self, split_ratio=0.8):

    # shuffling data
    indices = np.random.permutation(self.num_sample)

    start = int(split_ratio * self.num_sample)
    training_idx, test_idx = indices[:start], indices[start:]
    X_train, X_test = self.data[training_idx, :], self.data[test_idx, :]
    y_train, y_test = self.target[training_idx], self.target[test_idx]

    return X_train, y_train, X_test, y_test, training_idx, test_idx

def separateSampleClass(self):

    # Discriminating the classes sample
    self.ind_class = []
    for i in xrange(0, len(self.classes)):
        self.ind_class.append(np.where(self.target == self.classes[i]))

def getSampleClass(self):

    data = []
    target = []
    # Selecting the 'train sample' on the basis of the previously retrieved indices
    for i in xrange(0, len(self.classes)):
        data.append(self.data[self.ind_class[i]])
        target.append(self.target[self.ind_class[i]])

    return data, target

def getIndClass(self):

    return self.ind_class

