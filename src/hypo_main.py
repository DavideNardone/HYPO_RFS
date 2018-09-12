from __future__ import division

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

import numpy as np
np.set_printoptions(threshold=np.inf)
import sys
sys.path.insert(0, '/src')
import FeatureSelector as fs
import Hypo as hp
import Loader as lr


def main():

    ''' LOADING ANY DATASET '''
    dataset_dir = '/dataset'
    dataset_type = '/BIOLOGICAL'
    dataset_name = '/LUNG_DISCRETE'

    path_data_folder = dataset_dir + dataset_type + dataset_name
    path_data_file = path_data_folder + dataset_name

    variables = ['X', 'Y']
    # NB: If you get an error such as: 'Please use HDF reader for matlab v7.3 files',please change the 'format variable' to 'matlab_v73'
    D = lr.Loader(file_path=path_data_file,
                  format='matlab',
                  variables=variables,
                  name=dataset_name[1:]
                  ).getVariables(variables=variables)

    dataset = lr.Dataset(D['X'], D['Y'])

    dataset.standardizeDataset()

    X = dataset.data
    y = dataset.target

    # initializing feature selector parameters
    params = {

        'RFS':
            {
                'gamma': 1  # default value is 1
            }
    }

    # parameters to exhaustively combine
    tuned_parameters = {
        'gamma': [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10],
    }

    hypo = hp.Hypo(
        feature_selector = fs.FeatureSelector(name='RFS', params=params['RFS']),
        estimator = SVC(kernel="linear"),
        tuned_parameters = tuned_parameters,
        cv = 5,
        max_num_feat=100,
        scoring = 'accuracy'
    )

    hypo.fit(X, y)

    hypo.tuning_analysis(n_feats=10)

    hypo.plotAccuracy(
        n_feats = 100,
        step_size=5
    )

if __name__ == '__main__':
    main()