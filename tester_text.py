import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from explainer import Explainer


def read_csv(file_name):
    df = pd.read_csv(file_name)
    labels = df['-label-'].values.astype(np.bool)
    df.drop(['-label-', 'meta'], axis=1, inplace=True)
    features = np.array(list(df))
    data = np.asmatrix(df.values)
    return data, labels, features


def export_explanations(explanations, labels, scores, features):
    with open('files/explanations_text.csv', 'w', newline='') as export_file:
        writer = csv.writer(export_file)
        writer.writerow(["Observation", "Label", "Prediction", "Explanation"])
        for i_e, e_list in enumerate(explanations):
            if len(e_list) > 0:
                for explanation in e_list:
                    row = [i_e, labels[i_e], scores[i_e]] + list(features[explanation])
                    writer.writerow(row)
            else:
                row = [i_e, labels[i_e], scores[i_e], "No Explanation"]
                writer.writerow(row)


def main():
    input_file = "files/readyToGo.csv"
    data_file = "files/cache/data.pkl"
    labels_file = "files/cache/labels.pkl"
    features_file = "files/cache/features.pkl"
    model_file = "files/cache/model.pkl"
    try:
        data = pickle.load(open(data_file, "rb"))
        labels = pickle.load(open(labels_file, "rb"))
        features = pickle.load(open(features_file, "rb"))
    except IOError:
        data, labels, features = read_csv(input_file)
        pickle.dump(data, open(data_file, "wb"))
        pickle.dump(labels, open(labels_file, "wb"))
        pickle.dump(features, open(features_file, "wb"))
    try:
        model = pickle.load(open(model_file, "rb"))
    except IOError:
        print("Start fit")
        # model = LogisticRegression()
        model = RandomForestClassifier(random_state=0)
        model.fit(data, labels)
        print("Finish fitting")
        pickle.dump(model, open(model_file, "wb"))
    top_obs = 1000
    data = data[:top_obs, :]
    labels = labels[:top_obs]
    explainer = Explainer(model.predict_proba, np.zeros(data.shape[1]))
    explanations = explainer.explain(data)
    scores = model.predict_proba(data)[:, 1]
    export_explanations(explanations, labels, scores, features)


main()
