import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import sys
import argparse
from scipy import stats
from explainer import Explainer

sys.path.insert(0, r'/Users/xintianhan/Downloads/explanation/')
parser = argparse.ArgumentParser(description='mode or average operator for the continuous variable')
parser.add_argument('--opt', default='explanations_lc', help='File name.')
args = parser.parse_args()


def export_explanations(explanations, labels, scores, features, def_values, data, threshold, write, file_name):
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
    """
    f_path = './files/' + file_name + '.csv'
    with open(f_path, write, newline='') as export_file:
        writer = csv.writer(export_file)
        if write == 'w':
            writer.writerow(["Observation", "Label", "Prediction", "Threshold", "Explanation", "Change"])
        for i_e, e_list in enumerate(explanations):
            obs = data[i_e]
            if len(e_list) > 0:
                for e_ix, explanation in enumerate(e_list):
                    explanation.sort()
                    for f_ix in explanation:
                        change = features[f_ix]
                        change += " from "
                        change += str(obs[f_ix])
                        change += " to "
                        change += str(def_values[f_ix])
                        row = [i_e, labels[i_e], scores[i_e], threshold, e_ix + 1,  change]
                        writer.writerow(row)
            else:
                row = [i_e, labels[i_e], scores[i_e], threshold, 0, "No Explanation"]
                writer.writerow(row)


def main():
    # Load data; we only use training data for explanation
    X_train, y_train, X_test, y_test = pickle.load(open("./Data/LC_data.pickle", "rb"))
    feature_types = pickle.load(open("./Data/feature_types.pickle", "rb"))
    # save type of features
    feature_types = np.array(feature_types)
    # save name of features
    features = pickle.load(open("./Data/features.pickle", "rb"))
    # save category name for disrete values
    discrete_values = pickle.load(open("./Data/discrete_values.pickle", "rb"))
    #     input_file = "files/readyToGo.csv"
    #     data_file = "files/cache/data.pkl"
    #     labels_file = "files/cache/labels.pkl"
    #     features_file = "files/cache/features.pkl"
    model_file = "./files/cache/model2.pkl"
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
    # Prepare data
    for f_type in np.unique(feature_types):
        if f_type != -1:
            dummy_ixs = np.where(feature_types == f_type)[0]
            mode_ix = data[:, dummy_ixs].mean(axis=0).argsort()[-1]
            data = np.delete(data, mode_ix, 1)
            feature_types = np.delete(feature_types, mode_ix)
    # Default values
    def_values = np.empty(data.shape[1])
    cont_ixs = np.where(feature_types == -1)[0]
    def_values[cont_ixs] = data[:, cont_ixs].mean(axis=0)
    disc_ixs = np.where(feature_types != -1)[0]
    def_values[disc_ixs] = stats.mode(data[:, disc_ixs], axis=0)[0]
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
    explainer = Explainer(model.predict_proba, def_values)
    max_ite = 20
    export_f_name = args.opt
    explanations = explainer.explain(data, threshold, max_ite)
    scores = model.predict_proba(data)[:, 1]
    export_explanations(explanations, labels, scores, features, def_values, data, threshold, 'w', export_f_name)
    # explore different thresholds
    thresholds = [0.45, 0.4, 0.35, 0.3]
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        explainer = Explainer(model.predict_proba, def_values)
        max_ite = 20
        explanations = explainer.explain(data, threshold, max_ite)
        scores = model.predict_proba(data)[:, 1]
        export_explanations(explanations, labels, scores, features, def_values, data, threshold, 'a', export_f_name)


main()