import numpy as np
from scipy import stats
import bisect


class Explainer(object):
    def __init__(self, score_f, threshold=0.5, prune=True, omit_default=True, exp_return = False):
        # Scoring function of the model we want to explain
        self.score_f = score_f
        # Decision threshold of the model
        self.threshold = threshold
        # Whether we want to make sure that explanations are irreducible
        self.prune = prune
        # Whether we want to omit explanations from default decisions
        self.omit_default = omit_default
        # Whether we want to make decision based on default probability
        # or expected interest rate = probability of not default * interest rate
        self.exp_return = exp_return
    def explain(self, data, col_types, cat_groups=None, max_ite=20, num_of_obs = 1000, int_rates = None):
        """
Get explanations.
        :param data: Data
        :param col_types: Array with column types: 'c' for continuous and 'd' for discrete (i.e., categorical)
        :param cat_groups: List of categorical groups. Each element of the list is a list that includes the indices
        of the features that represent the same categorical variable.
        :param max_ite: integer, number of iterations to get explanations
        :param int_rates: interest rate; if provided, scores = (1-prob)*int_rates
        :return:    Element 1: List of explanations. Each element is a list that contains the explanations for the
        observation in the ith position. Each explanation is a list that contains the indices of the features present
        in the explanation.
                    Element 2: Default values. List of default values for each feature.
        """
        all_explanations = []
        # Get default values
        def_values = np.empty(data.shape[1])
        cont_ixs = col_types == 'c'
        cat_ixs = col_types == 'd'
        if any(cont_ixs):
            def_values[cont_ixs] = np.squeeze(np.asarray(np.average(data[:, cont_ixs], axis=0)))
        def_values[cat_ixs] = np.squeeze(stats.mode(data[:, cat_ixs])[0])
        # Add categorical variable defaults
        cat_defaults = {}
        if cat_groups is None:
            cat_groups = []
        for cat_group in cat_groups:
            max_ix = data[:, np.array(cat_group)].sum(axis=0).argmax()
            mode_ix = cat_group[max_ix]
            def_values[mode_ix] = 1
            for f in cat_group:
                cat_defaults[f] = mode_ix
        for iter in range(num_of_obs):
            obs = data[iter]
            obs = obs.reshape(1, -1)
            int_rate = None
            if self.exp_return:
                int_rate = int_rates[iter]
            if self.exp_return:
                score = (1-self.score_f(obs)[0, 1])*(100+int_rate) - self.threshold
            else:
                score = self.score_f(obs)[0, 1] - self.threshold
            # Get class of the observation
            class_val = 1 if score >= 0 else -1
            # Get relevant features to apply operators
            relevant_f = np.where(obs != def_values)[1]
            # Remove variables that are the "mode" of some categorical group
            relevant_f = relevant_f[np.in1d(relevant_f, list(cat_defaults.values()), invert=True)]
            # Keep track of all the relevant categorical columns
            relevant_cat_f = np.in1d(relevant_f, list(cat_defaults.keys()))
            # Keep track of all the modes of the relevant categorical columns
            cat_mode_f = np.array([cat_defaults[f] for f in relevant_f[relevant_cat_f]], dtype=int)
            # Set lists of explanations
            explanations = np.zeros((0, relevant_f.size))
            e_list = []
            if class_val == 1 or not self.omit_default:
                # Set first combination with no operators applied
                combs = [np.full(relevant_f.size, False, dtype=bool)]
                # Set list of scores
                scores = [score * class_val]
                for i in range(max_ite):
                    # Check if there are any more explanations
                    if not combs:
                        break
                    # Get next combination with the smallest score
                    comb = combs.pop(0)
                    score = scores.pop(0)
                    # Add to list of explanations if the class changed
                    if score < 0:
                        if self.prune:
                            comb = self.prune_explanation(obs, comb, def_values, relevant_f, cat_defaults, int_rate)
                        explanations = np.vstack((explanations, comb))
                        e_list.append(relevant_f[comb == 1].tolist())
                        # Add categorical modes
                        for f in relevant_f[comb]:
                            if f in cat_defaults:
                                e_list[-1].append(cat_defaults[f])
                        # add new prediction
                        e_list[-1] = [e_list[-1], self.threshold+score*class_val]
                    else:
                        # Get possible features to apply operator
                        active_f = np.where(np.logical_not(comb))[0]
                        # Build new possible combinations (one for each operator application)
                        new_combs = np.tile(comb, (active_f.size, 1))
                        new_combs[np.arange(active_f.size), active_f] = True
                        # Remove combinations that are a superset of an explanation.
                        matches = new_combs.dot(explanations.T) - explanations.sum(axis=1)
                        are_superset = np.unique(np.where(matches >= 0)[0])
                        new_combs = np.delete(new_combs, are_superset, axis=0)
                        if new_combs.shape[0] == 0:
                            continue
                        # Predict scores for new combs and add them to list
                        new_obs = np.tile(obs, (new_combs.shape[0], 1))
                        def_value_tiles = np.tile(def_values[relevant_f], (new_combs.shape[0], 1))
                        new_obs[:, relevant_f] = np.multiply(1 - new_combs, new_obs[:, relevant_f]) \
                                                 + np.multiply(new_combs, def_value_tiles)
                        # Set default value of categorical variables
                        new_obs[:, cat_mode_f] = new_combs[:, relevant_cat_f]
                        if self.exp_return:
                            new_scores = ((1-self.score_f(new_obs)) * (100+int_rate) - self.threshold)[:, 1] * class_val
                        else:
                            new_scores = (self.score_f(new_obs) - self.threshold)[:, 1] * class_val
                        for j, new_score in enumerate(new_scores):
                            ix = bisect.bisect(scores, new_score)
                            scores.insert(ix, new_score)
                            combs.insert(ix, new_combs[j, :])
            all_explanations.append(e_list)
            iter += 1
        return all_explanations, def_values

    def prune_explanation(self, obs, explanation, def_values, active_f, cat_defaults, int_rate):
        relevant_f = active_f[explanation]
        relevant_cat_f = np.in1d(relevant_f, list(cat_defaults.keys()))
        cat_mode_f = np.array([cat_defaults[f] for f in relevant_f[relevant_cat_f]], dtype=int)
        # Get number of explanation subsets (excluding all variables and no variables)
        n = 2 ** explanation.sum()
        combinations = range(1, n-1)
        # Remove powers of 2 (i.e., single feature combinations)
        combinations = [comb for comb in combinations if (comb & (comb - 1)) > 0]
        n = len(combinations)
        # Order by number of bits (i.e., try larger combinations first)
        combinations = sorted(combinations, key=lambda x: bin(x).count("1"), reverse=True)
        t_obs = np.matrix(obs, copy=True)
        i = 0
        if self.exp_return:
            score = (1 - self.score_f(obs)[0, 1]) * (100+int_rate) - self.threshold
        else:
            score = self.score_f(obs)[0, 1] - self.threshold
        class_val = 1 if score >= 0 else -1
        bits = 1 << np.arange(explanation.sum())
        while i < n:
            c = combinations[i]
            # Set features according to combination and predict
            e_bits = ((c & bits) > 0).astype(int)
            t_obs[:, relevant_f] = np.multiply(1 - e_bits, obs[:, relevant_f]) \
                                   + np.multiply(e_bits, def_values[relevant_f])
            t_obs[:, cat_mode_f] = e_bits[relevant_cat_f]
            if self.exp_return:
                score = ((1-self.score_f(obs)[0, 1])*(100+int_rate) - self.threshold) * class_val
            else:
                score = (self.score_f(obs)[0, 1] - self.threshold) * class_val
            if score < 0:
                # We have a shorter explanation
                explanation = np.in1d(active_f, relevant_f[e_bits == 1])
                # Keep only subsets of the combination that was found
                combinations = [comb for comb in combinations if (comb | c) <= c]
                i = 0
                n = len(combinations)
            i += 1
        return explanation
