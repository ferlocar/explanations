import csv
import numpy as np
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
from explainer import Explainer


def read_csv(file_name):
    rows = []
    metas = []
    labels = []
    with open(file_name, 'rb') as f_in:
        cin = csv.DictReader(f_in)
        row = None
        for row in cin:
            metas.append(row.pop('meta'))
            labels.append(bool(int(row.pop('-label-'))))
            rows.append(row.values())
        features = row.keys()
    data_type = np.float64
    data = np.matrix(rows, data_type)
    labels = np.array(labels, dtype=np.bool)
    features = np.array(features)
    return data, labels, metas, features


def export_explanations(explanations, data, labels, scores, features):
    with open('files/explanations.csv', 'wb') as export_file:
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
        data, labels, metas, features = read_csv(input_file)
        pickle.dump(data, open(data_file, "wb"))
        pickle.dump(labels, open(labels_file, "wb"))
        pickle.dump(features, open(features_file, "wb"))
    try:
        model = pickle.load(open(model_file, "rb"))
    except IOError:
        print("Start fit")
        # model = LogisticRegression()
        model = RandomForestClassifier()
        model.fit(data, labels)
        print("Finish fitting")
        pickle.dump(model, open(model_file, "wb"))
    top_obs = 1000
    data = data[:top_obs, :]
    labels = labels[:top_obs]
    explainer = Explainer(model.predict_proba, 0.5)
    explanations = explainer.explain(data)
    scores = model.predict_proba(data)[:, 1]
    export_explanations(explanations, data, labels, scores, features)


main()