import numpy as np
import bisect


class Explainer(object):
    def __init__(self, score_f, threshold=0.5, prune=True, omit_default=True):
        # Scoring function of the model we want to explain
        self.score_f = score_f
        # Decision threshold of the model
        self.threshold = threshold
        # Whether we want to make sure that explanations are irreducible
        self.prune = prune
        # Whether we want to omit explanations from default decisions
        self.omit_default = omit_default

    def explain(self, data, feature_types, max_ite=20, operator="set_average"):
        '''
        :param data: Data
        :param feature_types: continuous feature -1;
        categorical feature different integer for the same category starts from 0 and in an ascending order
        :param max_ite: integer, number of iterations to get explanations
        :param operator: string, explanation operators
        :return:
        '''
        all_explanations = []
        def_values = np.zeros(data.shape[1])
        # maximum integer of feature types
        maximum_feature_types = feature_types[-1]
        # number of continuous features
        num_of_continuous = sum(feature_types == -1)
        # number of discrete/categorical features
        num_of_discrete = maximum_feature_types + 1
        num_of_features = num_of_continuous + num_of_discrete
        # Init def_modes for discrete features
        def_modes = np.arange(num_of_discrete)
        indices_list = []
        # assign def_modes values: each def_modes saves the index which is the mode of the category
        for i in range(num_of_discrete):
            indices = np.where(feature_types == i)[0]
            indices_list.append(indices)
            max_count = 0
            for j in range(len(indices)):
                cur_count = sum(data[:,indices[j]])
                if cur_count > max_count:
                    max_count = cur_count
                    max_idx = indices[j]
            def_modes[i] = max_idx
        # Get default values
        if operator == "set_zero":
            # set zero for continoust valued but mode for categorical values
            def_values = np.zeros(data.shape[1])
        elif operator == "set_average":
            # set mean for continuous values and mode for categorical values
            def_values = np.squeeze(np.asarray(np.average(data, axis=0)))
        else:
            raise ValueError('Unsupported operator: {0}.'.format(operator))
        for obs in data:
            obs = obs.reshape(1, -1)
            score = self.score_f(obs)[0, 1] - self.threshold
            # Get class of the observation
            class_val = 1 if score >= 0 else -1
            # Get relevant features to apply operators (all features are relevant here)
            relevant_f = np.arange(num_of_features)
            # Set lists of explanations
            explanations = np.zeros((0, num_of_features))
            e_list = []
            if class_val == 1 or not self.omit_default:
                # Set first combination with no operators applied
                combs = [np.full(num_of_features, False, dtype=bool)]
                # Set list of scores
                scores = [score * class_val]
                for _ in range(max_ite):
                    # Check if there are any more explanations
                    if not combs:
                        break
                    # Get next combination with the smallest score
                    comb = combs.pop(0)
                    score = scores.pop(0)
                    # Add to list of explanations if the class changed
                    if score < 0:
                        if np.sum(comb) > 1:
#                             print(comb)
#                             print('prune')
                            if self.prune:
                                comb = self.prune_explanation(obs, comb, def_values, def_modes, indices_list, relevant_f)
#                             print(comb)
                        explanations = np.vstack((explanations, comb))
#                         print(relevant_f[comb == 1].tolist())
#                         print(comb)
#                         print(relevant_f.shape)
#                         print(relevant_f[comb == 1].tolist())
                        e_list.append(relevant_f[comb == 1].tolist())

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
                        active_f = np.delete(active_f, are_superset, axis=0)
                        if new_combs.shape[0] == 0:
                            continue
                        # Predict scores for new combs and add them to list
                        new_obs = np.tile(obs, (new_combs.shape[0], 1))
                        # def_value_tiles = np.tile(def_values[relevant_f], (new_combs.shape[0], 1))
                        # new_obs[:, relevant_f] = np.multiply(1 - new_combs, new_obs[:, relevant_f]) + \
                        #                          np.multiply(new_combs, def_value_tiles)
                        for k in range(active_f.size):
                            # set new value for continuous feature
                            if active_f[k] < num_of_continuous:
                                new_obs[k, active_f[k]] = def_values[active_f[k]]
                            # set new value for discrete feature
                            else:
                                cur_index = int(active_f[k] - num_of_continuous)
                                new_obs[k, indices_list[cur_index]] = 0
                                new_obs[k, def_modes[cur_index]] = 1
                        new_scores = (self.score_f(new_obs) - self.threshold)[:, 1] * class_val
                        for j, new_score in enumerate(new_scores):
                            ix = bisect.bisect(scores, new_score)
                            scores.insert(ix, new_score)
                            combs.insert(ix, new_combs[j, :])
            all_explanations.append(e_list)
        return all_explanations, def_values, def_modes

    def prune_explanation(self, obs, explanation, def_values, def_modes, indices_list, active_f):
        relevant_f = active_f[explanation]
        # Get number of explanation subsets (excluding all variables and no variables)
        n = 2 ** explanation.sum()
        combinations = range(1, n-1)
        # Remove powers of 2 (i.e., single feature combinations)
        combinations = [x for x in combinations if (x & (x - 1)) > 0]
        n = len(combinations)
        # Order by number of bits (i.e., try larger combinations first)
        combinations = sorted(combinations, key=lambda x: bin(x).count("1"), reverse=True)
        t_obs = np.matrix(obs, copy=True)
        i = 0
        score = self.score_f(obs)[0, 1] - self.threshold
        class_val = 1 if score >= 0 else -1
        bits = 1 << np.arange(explanation.sum())
        while i < n:
            c = combinations[i]
            # Set features according to combination and predict
            e_bits = ((c & bits) > 0).astype(int)
            for k in range(len(e_bits)):
                if e_bits[k] == 1:
                    # set new value for continuous feature
                    if relevant_f[k] < num_of_continuous:
                        t_obs[:, relevant_f[k]] = def_values[relevant_f[k]]
                    # set new value for discrete feature
                    else:
                        cur_index = int(relevant_f[k] - num_of_continuous)
                        t_obs[:, indices_list[cur_index]] = 0
                        t_obs[:, def_modes[cur_index]] = 1
            # t_obs[:, relevant_f] = np.multiply(1 - e_bits, obs[:, relevant_f]) + \
            #                         np.multiply(e_bits, def_values[relevant_f])
            score = (self.score_f(t_obs) - self.threshold)[0, 1] * class_val
            if score < 0:
                # We have a shorter explanation
                explanation = np.in1d(active_f, relevant_f[e_bits == 1])
                # Keep only subsets of the combination that was found
                combinations = [x for x in combinations if (x | c) <= c]
                i = 0
                n = len(combinations)
            i += 1
        return explanation