import mne
from mne.datasets import eegbci
from mne.time_frequency import psd_array_welch
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

subject = 1
train_runs = [3]
label_map = {"T0": 0, "T1": 1, "T2": 2}


class PSDTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, fmin_alpha=8, fmax_alpha=13, fmin_beta=13, fmax_beta=30
    ):
        self.fmin_alpha = fmin_alpha
        self.fmax_alpha = fmax_alpha
        self.fmin_beta = fmin_beta
        self.fmax_beta = fmax_beta

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for raw_chunk in X:
            data = raw_chunk.get_data()
            psd, freqs = psd_array_welch(
                data,
                sfreq=raw_chunk.info["sfreq"],
                fmin=8,
                fmax=30,
                n_fft=data.shape[1],
                n_per_seg=data.shape[1],
            )
            psd *= 1e12
            alpha_power = psd[
                :, (freqs >= self.fmin_alpha) & (freqs <= self.fmax_alpha)
            ].mean(axis=1)
            beta_power = psd[
                :, (freqs >= self.fmin_beta) & (freqs <= self.fmax_beta)
            ].mean(axis=1)
            features.append(np.concatenate([alpha_power, beta_power]))
        return np.array(features)


paths = eegbci.load_data(subject, train_runs)
chunks = []
labels = []

for run, path in zip(train_runs, paths):
    raw_run = mne.io.read_raw_edf(path, preload=True)
    sfreq = raw_run.info["sfreq"]

    annotations = raw_run.annotations
    for annot in annotations:
        onset = annot["onset"]
        duration = annot["duration"]
        label_desc = annot["description"]
        label_int = label_map.get(label_desc, 0)
        chunk = raw_run.copy().crop(tmin=onset, tmax=onset + duration)
        chunks.append(chunk)
        labels.append(label_int)


pipeline = Pipeline(
    [
        ("psd", PSDTransformer()),
        ("pca", PCA(n_components=5)),
        ("clf", SVC(kernel="linear")),
    ]
)

pipeline.fit(chunks, labels)
joblib.dump(pipeline, "eeg_pipeline.pkl")
