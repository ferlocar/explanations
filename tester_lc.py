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
from explainer import Explainer

sys.path.insert(0, r'/Users/xintianhan/Downloads/explanation/')
parser = argparse.ArgumentParser(description='mode or average operator for the continuous variable')
# parser.add_argument('--opt', default='explanations_lc', help='File name.')
parser.add_argument('--model', default='lc', help='Model name: lc, rf, nb, bag, dt, svm, gb, rg')
parser.add_argument('--exp', action='store_true', help='use expected interest rate score or not')
# parser.add_argumnet('--grade', default='A', help='use only grade A or all the grades: A or all')
args = parser.parse_args()
args_model = args.model
args_exp = args.exp
# threshold for people getting credits
upper_threshold = 0.5
upper_threshold_exp = 105
# threshold for people not getting credits
lower_threshold = 0.05
lower_threshold_exp = 65
def export_explanations(explanations, labels, scores, features, def_values, data, threshold, write, file_name, mins, maxs, grades_train):
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
                for e_ix, exp_score in enumerate(e_list):
                    # exp_score[0] saves explanation; exp_score[1] saves score after removing evidence
                    explanation = exp_score[0]
                    score = exp_score[1]
                    explanation.sort()
                    for f_ix in explanation:
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
    X_train, y_train, X_test, y_test = pickle.load(open("./Data/LC_data.pickle", "rb"))
    feature_types = pickle.load(open(".//Data/feature_types.pickle", "rb"))
    # save type of features
    feature_types = np.array(feature_types)
    # save name of features
    features = pickle.load(open("./Data/features.pickle", "rb"))
    # save category name for disrete values
    discrete_values = pickle.load(open("./Data/discrete_values.pickle", "rb"))
    # save grade for each instance in training set
    grades_test = pickle.load(open("./Data/grades_test.pickle", "rb"))
    # save interest rate for each instance in training set
    int_rates_test = pickle.load(open("./Data/int_rates_test.pickle", "rb"))
    # # set for grade that has A
    # A_set = (grades_test == 'A')
    # save an array of mins of continuous features; used for transform the scaled continuous features back
    test_mins = pickle.load(open("./Data/test_mins.pickle", "rb"))
    # save an array of maxs of continuous features; used for transform the scaled continuous features back
    test_maxs = pickle.load(open("./Data/test_maxs.pickle", "rb"))
    #     input_file = "files/readyToGo.csv"
    #     data_file = "files/cache/data.pkl"
    #     labels_file = "files/cache/labels.pkl"
    #     features_file = "files/cache/features.pkl"
    model_file = "./files/cache/model2pkl_"+args_model
    #     try:
    #         data = pickle.load(open(data_file, "rb"))
    #         labels = pickle.load(open(labels_file, "rb"))
    #         features = pickle.load(open(features_file, "rb"))
    #     except IOError:
    #         data, labels, features = read_csv(input_file)
    #         pickle.dump(data, open(data_file, "wb"))
    #         pickle.dump(labels, open(labels_file, "wb"))
    #         pickle.dump(features, open(features_file, "wb"))
    # Update feature names
    f_names = []
    f_ix = 0
    for f in features:
        f_type = feature_types[f_ix]
        if f_type == -1:
            f_names.append(f)
            f_ix += 1
        else:
            f_values = discrete_values[f_type]
            for f_value in f_values:
                f_names.append(f + "::" + f_value)
                f_ix += 1
    features = f_names
    data = X_train
    labels = y_train
    features = np.array(features)
    try:
        model = pickle.load(open(model_file, "rb"))
    except IOError:
        print("Start fit")
        if args_model == 'lc':
            model = LogisticRegression(penalty='l1', C=10000.0)
        elif args_model == 'rf':
            model = RandomForestClassifier(min_samples_leaf = 8, n_estimators = 35)
        elif args_model == 'nb':
            model = GaussianNB()
        #### Not Complete
        model.fit(data, labels)
        print("Finish fitting")
        pickle.dump(model, open(model_file, "wb"))
    # Prepare categorical values
    col_types = np.empty(feature_types.size, dtype=str)
    cat_groups = list()
    col_types[np.where(feature_types == -1)[0]] = "cont"
    col_types[np.where(feature_types != -1)[0]] = "disc"
    cat_n = feature_types.max()
    for i in range(cat_n + 1):
        cat_groups.append(list(np.where(feature_types == i)[0]))
    # if args.grade == 'A':
    #     data = data[Aset,:]
    #     labels = labels[Aset]
    #     grades_train = grades_train[Aset]
    data = X_test
    labels = y_test
    if args_exp:
        scores = (1-model.predict_proba(data)[:, 1]) * (100+int_rates_test)
        print('greater than 1.05:', sum(scores > upper_threshold_exp)/20000.0)
        print('smaller than 0.65:', sum(scores < lower_threshold_exp)/20000.0)
    else:
        scores = model.predict_proba(data)[:, 1]
        print('greater than 0.5:', sum(scores > upper_threshold)/20000.0)
        print('smaller than 0.05:', sum(scores < lower_threshold)/20000.0)
    # top_obs = 2000
    # data = data[:top_obs, :]
    # labels = labels[:top_obs]
    num_of_obs = 1000
    if args_exp:
        threshold = upper_threshold_exp
    else:
        threshold = upper_threshold
    explainer = Explainer(model.predict_proba, threshold, exp_return = args_exp)
    max_ite = 20
    export_f_name = 'explanations_'+args_model
    if args_exp:
        export_f_name += '_exp'
    explanations, def_values = explainer.explain(data, col_types, cat_groups, max_ite, num_of_obs, int_rates_test)
    if args_exp:
        scores = (1-model.predict_proba(data)[:, 1]) * (100+int_rates_test)
    else:
        scores = model.predict_proba(data)[:, 1]
    export_explanations(explanations, labels, scores, features, def_values, data, threshold, 'w', export_f_name, test_mins, test_maxs, grades_test)
    # explore different thresholds
    if args_exp:
        thresholds = [lower_threshold_exp]
    else:
        thresholds = [lower_threshold]
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        explainer = Explainer(model.predict_proba, threshold, omit_default = False, exp_return = args_exp)
        max_ite = 20
        explanations, def_values = explainer.explain(data, col_types, cat_groups, max_ite, num_of_obs, int_rates_test)
        if args_exp:
            scores = (1 - model.predict_proba(data)[:, 1]) * (100+int_rates_test)
        else:
            scores = model.predict_proba(data)[:, 1]
        export_explanations(explanations, labels, scores, features, def_values, data, threshold, 'a', export_f_name, test_mins, test_maxs, grades_test)


main()