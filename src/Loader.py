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
