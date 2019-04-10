import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
import pickle
import sys
import argparse
from scipy import stats
from explainer import Explainer

'''
New Interface
User input data and def values
'''
sys.path.insert(0, r'/Users/xintianhan/Downloads/explanation/')
parser = argparse.ArgumentParser(description='mode or average operator for the continuous variable')
# parser.add_argument('--opt', default='explanations_lc', help='File name.')
parser.add_argument('--model', default='lc', help='Model name: lc, rf, nb, bag, dt, svm, gb, rg')
parser.add_argument('--exp', action='store_true', help='use expected interest rate score or not')
# parser.add_argumnet('--grade', default='A', help='use only grade A or all the grades: A or all')
args = parser.parse_args()
args_model = args.model
args_exp = args.exp
# Global Variable Threshold
# threshold for people getting credits
upper_threshold = 0.5
upper_threshold_exp = 105
# threshold for people not getting credits
lower_threshold = 0.05
lower_threshold_exp = 65
def export_explanations(explanations, labels, scores, features, def_values, data, threshold, write, file_name, mins,
                        maxs, grades_train, scoring_function):
    """
Export explanations to csv file.
    :param explanations: list of explanation for each observation;
            explanation: list of group of feature indices; each group is one 'explanation' for the observation
    :param labels: label for the observation
    :param scores: predicted scores for the observation
    :param features: names of each feature
    :param def_values: list of default values for each feature
    :param data: original data
    :param threshold: threshold used to make the decision
    :param write: overwrite 'w'; write in addition to the current file 'a'
    :param file_name: name of the file to which explanations are exported
    :param mins: an array of mins of continuous features; used for transform the scaled continuous features back
    :param maxs: an array of maxs of continuous features; used for transform the scaled continuous features back
    :param grades_train: grade for each instance
    """
    f_path = './files/' + file_name + '.csv'
    with open(f_path, write, newline='') as export_file:
        writer = csv.writer(export_file)
        if write == 'w':
            writer.writerow(["Observation", "Grade", "Label", "Prediction", "Threshold", "New Prediction", "Explanation", "Change"])
        for i_e, e_list in enumerate(explanations):
            obs = data[i_e]
            if len(e_list) > 0:
                for e_ix, explanation in enumerate(e_list):
                    # exp_score[0] saves explanation; exp_score[1] saves score after removing evidence
                    obs_copy = obs.copy().reshape(1, -1)
                    obs_copy[0, explanation] = def_values[explanation]
                    score = scoring_function(obs_copy)
                    explanation.sort()
                    for f_ix in explanation:
                        # print(f_ix)
                        change = features[f_ix]
                        change += " from "
                        org_obs = obs[f_ix]*(maxs[f_ix]-mins[f_ix]) + mins[f_ix]
                        change += str(org_obs)
                        change += " to "
                        org_def_values = def_values[f_ix]*(maxs[f_ix]-mins[f_ix]) + mins[f_ix]
                        change += str(org_def_values)
                        row = [i_e, grades_train[i_e], labels[i_e], scores[i_e], threshold, score, e_ix + 1,  change]
                        writer.writerow(row)
            else:
                row = [i_e, grades_train[i_e], labels[i_e],  scores[i_e],  threshold, None, 0, "No Explanation"]
                writer.writerow(row)


def main():
    # Load data; we only use training data for explanation
    _, y_train, _, y_test = pickle.load(open("./Data/LC_data.pickle", "rb"))
    # def_values: explainer changes current feature to default values
    def_values = pickle.load(open("./Data/def_values.pickle", "rb"))
    # save grade for each instance in test set
    grades_test = pickle.load(open("./Data/grades_test.pickle", "rb"))
    # # save interest rate for each instance in test set
    int_rates_test = pickle.load(open("./Data/int_rates_test.pickle", "rb"))
    # save an array of mins of continuous features; used for transform the scaled continuous features back
    test_mins = pickle.load(open("./Data/test_mins.pickle", "rb"))
    # save an array of maxs of continuous features; used for transform the scaled continuous features back
    test_maxs = pickle.load(open("./Data/test_maxs.pickle", "rb"))
    # file name of the model
    model_file = "./files/cache/model2_{0}.pkl".format(args_model)
    # load feature names without mode
    f_names = pickle.load(open("./Data/f_names.pickle", "rb"))
    # load X_train, X_test without mode
    X_train = pickle.load(open("./Data/X_train_no_mode.pickle", "rb"))
    X_test = pickle.load(open("./Data/X_test_no_mode.pickle", "rb"))

    try:
        model = pickle.load(open(model_file, "rb"))
    except IOError:
        print("Start fit")
        if args_model == 'rf':
            model = RandomForestClassifier(min_samples_leaf = 8, n_estimators = 35)
        elif args_model == 'nb':
            model = GaussianNB()
        else:
            # Default model is a logistic regression
            model = LogisticRegression(penalty='l1', C=10000.0)
        #### Not Complete
        model.fit(X_train, y_train)
        print("Finish fitting")
        pickle.dump(model, open(model_file, "wb"))
    data = X_test
    labels = y_test
    if args_exp:
        threshold = upper_threshold_exp
    else:
        threshold = upper_threshold

    def expected_int(df):
        return (1 - model.predict_proba(df)[:, 1]) * (100 + int_rates_test)

    def get_prob(df):
        return model.predict_proba(df)[:, 1]

    scoring_function = expected_int if args_exp else get_prob

    explainer = Explainer(scoring_function, def_values)
    max_ite = 20
    export_f_name = 'explanations_'+args_model
    if args_exp:
        export_f_name += '_exp'
    explanations = explainer.explain(data, threshold, max_ite)
    if args_exp:
        scores = scoring_function(data)
    else:
        scores = model.predict_proba(data)[:, 1]

    print('Explain threshold:', threshold)

    export_explanations(explanations, labels, scores, f_names, def_values, data, threshold, 'w',
                        export_f_name, test_mins, test_maxs, grades_test, scoring_function)
    # explore different thresholds
    if args_exp:
        thresholds = [lower_threshold_exp]
    else:
        thresholds = [lower_threshold]
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        explainer = Explainer(scoring_function, def_values)
        max_ite = 20
        explanations = explainer.explain(data, threshold, max_ite)
        if args_exp:
            scores = expected_int(data)
        else:
            scores = model.predict_proba(data)[:, 1]
        print('Explain threshold:', threshold)
        export_explanations(explanations, labels, scores, f_names, def_values, data, threshold, 'a', export_f_name,
                            test_mins, test_maxs, grades_test, scoring_function)


main()
