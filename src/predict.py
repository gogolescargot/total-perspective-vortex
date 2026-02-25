from joblib import load
import numpy as np
from train import load_data, EXPERIMENTS


def predict_epochs(model_path, X, y):
    clf = load(model_path)

    scores = []
    print("epoch_nb = [prediction] [truth] equal?")
    print("---------------------------------------------")
    for n in range(X.shape[0]):
        pred = clf.predict(X[n : n + 1, :, :])[0]
        truth = y[n]
        print(
            f"epoch {n:2} = [{pred}] [{truth}] \
            {'' if pred == truth else False}"
        )
        scores.append(1 - abs(pred - truth))

    return float(np.mean(scores).round(3))


def predict_all_experiments(model_path, path=None):
    all_accs = []
    for exp_idx in range(len(EXPERIMENTS)):
        print(f"\nExperiment {exp_idx + 1} (runs {EXPERIMENTS[exp_idx]})")
        epochs, _ = load_data(None, None, exp_idx, data_path=path)

        if len(epochs) == 0:
            print("No epochs found, skipping.")
            continue

        X = epochs.get_data()
        y = (epochs.events[:, -1] == 2).astype(int)

        clf = load(model_path)
        scores = clf.predict(X)
        acc = float(np.mean(scores == y).round(3))
        print(f"Accuracy: {acc:.4f}")
        all_accs.append(acc)

    if all_accs:
        print(
            f"\nTotal mean accuracy across all experiments: {np.mean(all_accs):.4f}"
        )
    else:
        print("No results to aggregate.")


def predict(subjects, runs, experiment, model_path, path=None):
    if subjects is None and runs is None and experiment is None:
        predict_all_experiments(model_path, path)
        return

    epochs, _ = load_data(subjects, runs, experiment, data_path=path)

    if len(epochs) == 0:
        print("No epochs found after preprocessing, aborting.")
        return

    X = epochs.get_data()
    y = (epochs.events[:, -1] == 2).astype(int)

    mean_acc = predict_epochs(model_path, X, y)
    print(f"Mean accuracy: {mean_acc}")
