import csv
import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import sys
sys.path.insert(0, r'/Users/xintianhan/Downloads/explanation/')
import argparse
from explainer_new import Explainer
parser = argparse.ArgumentParser(description='zero or average operator for the continuous variable')
parser.add_argument('--opt', default = 'set_zero', help = 'set_zero or set_average')
args = parser.parse_args()
def export_explanations(explanations, labels, scores, features, def_values, def_modes, data, discrete_values,
                        feature_types, threshold, write, file_name):
    '''

    :param explanations: list of explanation for each observation;
            explanation: list of group of feature indices; each group is one 'explanation' for the observation
    :param labels: label for the observation
    :param scores: predicted scores for the observation
    :param features: names of each feature
    :param def_values: list of continuous values that the operator set to the feature
    :param def_modes: list of indices denotes the mode of the category variable
    :param data: original data
    :param discrete_values: values of the category variable
    :param feature_types: np.array, types of feature; -1 for continuous values; different category has different values
    :param threshold: threshold used to make the decision
    :param write: overwrite 'w'; write in addition to the current file 'a'
    :param file_name: which file write in
    :return:
    '''
    with open('../files/' + file_name + '.csv', write, newline='') as export_file:
        writer = csv.writer(export_file)
        if write == 'w':
            writer.writerow(["Observation", "Label", "Prediction", "Threshold", "Explanation"])
        for i_e, e_list in enumerate(explanations):
            if len(e_list) > 0:
                for explanation in e_list:
                    row_features = list(features[explanation])
                    row_detail_exp = []
                    for i in range(len(row_features)):
                        temp = features[explanation[i]]
                        data_temp = data[i_e].copy()
                        # extract row feature values for category values
                        if explanation[i] >= len(features) - len(def_modes):
                            # idx of discrete features
                            idx_row = explanation[i] - (len(features) - len(def_modes))
                            temp += '::'
                            idx_feat = (feature_types == idx_row)
                            temp += discrete_values[idx_row][np.where(data_temp[idx_feat] == 1)[0].item()]
                            temp += '::'
                            temp += discrete_values[idx_row][
                                def_modes[idx_row] - np.where(feature_types == idx_row)[0][0].item()]
                        # extract row feature values for continuouse values
                        else:
                            temp += '::'
                            temp += str(data_temp[explanation[i]])
                            temp += '::'
                            temp += str(def_values[explanation[i]])
                        row_detail_exp.append(temp)

                    row = [i_e, labels[i_e], scores[i_e], threshold] + row_detail_exp
                    writer.writerow(row)
            else:
                row = [i_e, labels[i_e], scores[i_e], threshold, "No Explanation"]
                writer.writerow(row)


def main():
    #Load data; we only use training data for explanation
    X_train, y_train, X_test, y_test = pickle.load(open("../Data/LC_data.pickle", "rb"))
    feature_types = pickle.load(open("../Data/feature_types.pickle", "rb"))
    # save type of features
    feature_types = np.array(feature_types)
    # save name of features
    features = pickle.load(open("../Data/features.pickle", "rb"))
    # save category name for disrete values
    discrete_values = pickle.load(open("../Data/discrete_values.pickle", "rb"))
    #     input_file = "files/readyToGo.csv"
    #     data_file = "files/cache/data.pkl"
    #     labels_file = "files/cache/labels.pkl"
    #     features_file = "files/cache/features.pkl"
    model_file = "../files/cache/model.pkl"
    #     try:
    #         data = pickle.load(open(data_file, "rb"))
    #         labels = pickle.load(open(labels_file, "rb"))
    #         features = pickle.load(open(features_file, "rb"))
    #     except IOError:
    #         data, labels, features = read_csv(input_file)
    #         pickle.dump(data, open(data_file, "wb"))
    #         pickle.dump(labels, open(labels_file, "wb"))
    #         pickle.dump(features, open(features_file, "wb"))
    data = X_train
    labels = y_train
    features = np.array(features)
    try:
        model = pickle.load(open(model_file, "rb"))
    except IOError:
        print("Start fit")
        # model = LogisticRegression()
        model = LogisticRegression(penalty='l1', C=4.641588833612778)
        model.fit(data, labels)
        print("Finish fitting")
        pickle.dump(model, open(model_file, "wb"))
    top_obs = 1000
    data = data[:top_obs, :]
    labels = labels[:top_obs]
    threshold = 0.5
    explainer = Explainer(model.predict_proba, threshold)
    max_ite = 20
    operator = args.opt
    explanations, def_values, def_modes = explainer.explain(data, feature_types, max_ite, operator)
    scores = model.predict_proba(data)[:, 1]
    export_explanations(explanations, labels, scores, features, def_values, def_modes, data, discrete_values,
                        feature_types, threshold, 'w', operator)
    # explore different thresholds
    thresholds = [0.45, 0.4, 0.35, 0.3]
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        explainer = Explainer(model.predict_proba, threshold)
        max_ite = 20
        explanations, def_values, def_modes = explainer.explain(data, feature_types, max_ite, operator)
        scores = model.predict_proba(data)[:, 1]
        export_explanations(explanations, labels, scores, features, def_values, def_modes, data, discrete_values,
                            feature_types, threshold, 'a', operator)


main()