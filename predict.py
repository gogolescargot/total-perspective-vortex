from joblib import load
import numpy as np
from train import load_data


def predict_epochs(model_path, X, y):
    try:
        clf = load(model_path)
    except FileNotFoundError:
        raise Exception(f"File not found: {model_path}")

    scores = []
    print("epoch_nb =  [prediction]    [truth]    equal?")
    print("---------------------------------------------")
    for n in range(X.shape[0]):
        pred = clf.predict(X[n : n + 1, :, :])[0]
        truth = y[n]
        print(
            f"epoch {n:2} =      [{pred}]           [{truth}]      {'' if pred == truth else False}"
        )
        scores.append(1 - abs(pred - truth))

    return float(np.mean(scores).round(3))


def predict(subjects, runs, experiment, model_path):
    epochs, _ = load_data(subjects, runs, experiment)

    if len(epochs) == 0:
        print("No epochs found after preprocessing, aborting.")
        return

    X = epochs.get_data()
    y = (epochs.events[:, -1] == 2).astype(int)
    print((epochs.events[:, -1] == 2).astype(int))

    mean_acc = predict_epochs(model_path, X, y)
    print(f"Mean accuracy: {mean_acc}")
