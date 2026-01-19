import mne
from mne.datasets import eegbci
from train import PSDTransformer
import numpy as np
import joblib

subject = 1
test_runs = [4]
chunk_duration = 2.0
labels_map = {0: "T0", 1: "T1", 2: "T2"}

pipeline = joblib.load("eeg_pipeline.pkl")
print("Pipeline chargé")

paths = eegbci.load_data(subject, test_runs)

for run, path in zip(test_runs, paths):
    raw_run = mne.io.read_raw_edf(path, preload=True)
    sfreq = raw_run.info["sfreq"]
    n_samples_chunk = int(chunk_duration * sfreq)
    n_samples_run = raw_run.n_times

    starts = np.arange(0, n_samples_run, n_samples_chunk)
    for i, s in enumerate(starts):
        e = min(s + n_samples_chunk, n_samples_run)
        chunk = raw_run.copy().crop(
            tmin=s / sfreq, tmax=min(e / sfreq, raw_run.times[-1])
        )

        y_pred = pipeline.predict([chunk])

        print(
            f"Run {run}, Chunk {i} ({s / sfreq:.1f}-{e / sfreq:.1f}s) → predicted: {labels_map.get(y_pred[0], y_pred[0])}"
        )
