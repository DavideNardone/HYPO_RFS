from __future__ import division

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

import numpy as np
np.set_printoptions(threshold=np.inf)
import sys
sys.path.insert(0, '/src')
import FeatureSelector as fs
import Hypo as hp


def main():

    # loading data
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # initializing feature selector parameters
    params = {

        'LASSO':
            {
                'alpha': 1  # default value is 1
            }
    }

    # parameters to exhaustively combine
    tuned_parameters = {

        'alpha': [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10],
        # 'delta': [10,20],

    }

    hypo = hp.Hypo(
        feature_selector=fs.FeatureSelector(name='LASSO', params=params['LASSO']),
        estimator=SVC(kernel="linear"),
        tuned_parameters=tuned_parameters,
        cv=5,
        max_num_feat=25,
        scoring='accuracy'
    )

    hypo.fit(X, y)
    hypo.tuning_analysis(n_feats=10)

if __name__ == '__main__':
    main()