import mne
from mne.datasets import eegbci
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from csp import CSP
from sklearn.preprocessing import StandardScaler
import joblib

mne.set_log_level("WARNING")
logging.basicConfig(level=logging.WARNING)

EXPERIMENTS = [
    [3, 7, 11],
    [4, 8, 12],
    [5, 9, 13],
    [6, 10, 14],
    [3, 4, 7, 8, 11, 12],
    [5, 6, 9, 10, 13, 14],
]


def preprocess_raw(raw, l_freq=7.0, h_freq=30.0, sfreq=160):
    if abs(raw.info.get("sfreq", 0) - sfreq) > 1e-8:
        raw.resample(sfreq, npad="auto")

    eegbci.standardize(raw)

    raw.filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge")

    return raw


def load_epochs(raw, tmin=0.5, tmax=3.5):
    events, _ = mne.events_from_annotations(raw)

    raw.set_annotations(None)

    valid_events = []
    for event in events:
        if event[2] in [1, 2]:
            valid_events.append(event)

    if len(valid_events) == 0:
        return []

    events_array = np.array(valid_events)

    picks = mne.pick_types(
        raw.info,
        eeg=True,
        meg=False,
        stim=False,
        eog=False,
        exclude="bads",
    )

    epochs = mne.Epochs(
        raw,
        events_array,
        event_id={"T1": 1, "T2": 2},
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    return epochs


def load_data(
    subjects,
    runs,
    experiment,
    tmin=0.5,
    tmax=3.5,
    l_freq=7.0,
    h_freq=30.0,
    sfreq=160,
):
    epochs_list = []
    subjects_list = []

    if experiment is None and runs is None:
        runs = list(range(3, 15))
    elif experiment is not None:
        runs = EXPERIMENTS[experiment]

    if 1 in runs or 2 in runs:
        raise ValueError("Runs 1 and 2 are not valid for motor imagery tasks")

    for subject in subjects:
        paths = eegbci.load_data(subject, runs)
        for path in paths:
            raw = mne.io.read_raw_edf(path, preload=False)
            raw.load_data()

            raw = preprocess_raw(raw, l_freq, h_freq, sfreq)

            epochs = load_epochs(raw, tmin, tmax)

            if len(epochs) > 0:
                epochs_list.append(epochs)
                subjects_list.extend([subject] * len(epochs))

            raw.close()
            del raw

    if not epochs_list:
        raise RuntimeError("No epochs found for provided subjects/runs")

    all_epochs = mne.concatenate_epochs(epochs_list)
    all_subjects = np.array(subjects_list)

    return all_epochs, all_subjects


def train(subjects, runs, experiment, out):
    epochs, _ = load_data(
        subjects, runs, experiment, tmin=0.5, tmax=3.5, l_freq=8.0, h_freq=30.0
    )

    if len(epochs) == 0:
        print("No epochs found after preprocessing, aborting.")
        return

    X = epochs.get_data()
    y = epochs.events[:, 2]
    y = (y == 2).astype(int)

    pipeline = Pipeline(
        [
            ("csp", CSP(n_components=4, reg=None)),
            ("scaler", StandardScaler()),
            ("lda", LDA(solver="lsqr", shrinkage="auto")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
        "csp__n_components": [2, 4, 6],
        "csp__reg": [None, 0.01, 0.1],
        "lda__shrinkage": [None, "auto"],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    print("Running GridSearchCV on training data...")
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"Test set accuracy (best model): {test_score:.4f}")

    cross_val_scores = cross_val_score(best_model, X, y, cv=cv, n_jobs=-1)
    print(f"Cross-validation scores (best model): {cross_val_scores}")

    joblib.dump(best_model, out)
    print(f"Model saved to {out}")
