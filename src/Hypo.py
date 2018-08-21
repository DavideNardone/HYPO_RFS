from sklearn.model_selection import KFold
import itertools
import numpy as np

class Hypo:

    def __init__(
            self,
             feature_selector,
             estimator,
             tuned_parameters,
             cv=5,
             max_num_feat=300,
             scoring='accuracy'
    ):

        self.feature_selector = feature_selector
        self.estimator = estimator
        self.tuned_parameters = tuned_parameters
        self.cv = cv
        self.max_num_feat = max_num_feat
        self.scoring = scoring
        self.CMB = {}

    def create_grid(self):

        comb = []
        params_name = []

        for name, tun_par in self.tuned_parameters.iteritems():
            comb.append(tun_par)
            params_name.append(name)

        combs =[]
        for t in itertools.product(*comb):
            combs.append(t)

        return combs, params_name

    def fit(self,X,y):

        # create all combinations parameters
        combs,params_name = self.create_grid()

        n_iter = 1
        for comb in combs:

            CV = np.ones([self.cv, self.max_num_feat]) * 0
            self.CMB.update({comb: {}})

            print ('Computing ' + n_iter.__str__() + '-th combination...')

            # set i-th parameters combination parameters for the current feature selector
            self.feature_selector.setParams(comb, params_name)

            cc_fold = 0
            kf = KFold(n_splits=self.cv)
            for train_index, test_index in kf.split(X):

                kth_scores = []

                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                idx = self.feature_selector.fit(X_train, y_train)

                # classification step on the first max_num_feat
                for n_rep in xrange(1, self.max_num_feat + 1, 1):

                    X_train_fs = X_train[:, idx[0:n_rep]]
                    X_test_fs = X_test[:, idx[0:n_rep]]

                    # Training the algorithm using the selected predictors and target.
                    self.estimator.fit(X_train_fs, y_train)
                    # Record error for testing
                    _score = self.estimator.score(X_test_fs, y_test)

                    kth_scores.append(_score)  # it contains the max_num_feat scores for the k-th CV fold

                CV[cc_fold, :] = kth_scores
                cc_fold += 1

            n_iter += 1
            avg_scores = np.mean(CV, axis=0)
            std_scores = np.std(CV, axis=0)

            self.CMB[comb]['ACC'] = avg_scores
            self.CMB[comb]['STD'] = std_scores

    def tuning_analysis(self, n_feats):

        voting_matrix = {}
        _res_voting = {}

        min_var = 99999999
        min_hyp_par = {}

        combs = self.CMB.keys()
        combs.sort()

        for comb in combs:

            voting_matrix[comb] = np.zeros([1, n_feats])
            value = self.CMB[comb]
            curr_var = np.var(value['ACC'])
            if curr_var < min_var:
                min_var = curr_var
                min_hyp_par = comb

            print 'Hyper-params. comb=%s has minimum variance of %s' % (min_hyp_par, min_var)

        combs = self.CMB.keys()
        combs.sort()

        print '\nApplying majority voting...'
        for j in xrange(0, n_feats):
            _competitors = {}
            for comb in combs:
                _competitors[comb] = self.CMB[comb]['ACC'][j]

            # getting the winner accuracy for all the combinations computed
            winners = [comb for m in [max(_competitors.values())] for comb, val in _competitors.iteritems() if
                       val == m]
            for winner in winners:
                voting_matrix[winner][0][j] = 1

        # getting the parameter with largest voting
        for comb in combs:
            _res_voting[comb] = np.sum(voting_matrix[comb][0])

        _max = -9999999
        best_comb = {}
        for comb in combs:
            if _res_voting[comb] > _max:
                _max = _res_voting[comb]
                best_comb = comb
            print ('Parameters set: ' + comb.__str__() + ' got votes: ' + _res_voting[comb].__str__())

        print ('\nBest parameters set found on development set is: ' + best_comb.__str__())

        return best_comb