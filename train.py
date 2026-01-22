import mne
from mne.datasets import eegbci
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
import joblib

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
        runs = list(range(1, 15))
    elif experiment is not None:
        runs = EXPERIMENTS[experiment]

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
            ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
            ("scaler", StandardScaler()),
            ("lda", LDA(solver="lsqr", shrinkage="auto")),
        ]
    )

    param_grid = {
        "csp__n_components": [4, 6, 8],
        "csp__reg": [None, 0.1, 0.5],
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_inner,
        n_jobs=-1,
        scoring="accuracy",
        verbose=0,
        refit=True,
    )

    gs.fit(X_train, y_train)
    print(f"Best params (from inner CV): {gs.best_params_}")
    print(f"Best CV score (inner): {gs.best_score_:.3f}")

    test_score = gs.best_estimator_.score(X_test, y_test)
    print(f"Held-out test score: {test_score:.3f}")

    final_model = pipeline.set_params(**gs.best_params_)
    final_model.fit(X, y)
    cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(final_model, X, y, cv=cv_final, n_jobs=-1)
    print(
        f"CV score final (refit on all data): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
    )

    joblib.dump(final_model, out)
    print(f"Modèle sauvegardé dans {out}")
