import mne
from mne.datasets import eegbci
import matplotlib.pyplot as plt
from train import L_FREQ, H_FREQ


def visualize(subjects, runs, path=None):
    subject = subjects
    runs = runs

    paths = eegbci.load_data(subject, runs, path=path)
    raws = [mne.io.read_raw_edf(p, preload=True) for p in paths]
    raw = mne.concatenate_raws(raws)
    raw.plot(title="Raw signal")

    raw.notch_filter(50.0, fir_design="firwin")
    raw_pp = raw.copy().filter(
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        fir_design="firwin",
        skip_by_annotation="edge",
    )
    raw_pp.plot(title=f"Preprocessed signal ({L_FREQ}-{H_FREQ} Hz bandpass)")
    plt.show()
