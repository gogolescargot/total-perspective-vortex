import mne
from mne.datasets import eegbci
import matplotlib.pyplot as plt


def visualize(subjects, runs, path=None):
    subject = subjects
    runs = runs

    paths = eegbci.load_data(subject, runs, path=path)
    raws = [mne.io.read_raw_edf(p, preload=True) for p in paths]
    raw = mne.concatenate_raws(raws)
    raw.plot(title="Raw signal")

    raw.notch_filter(50.0, fir_design="firwin")
    raw_pp = raw.copy().filter(
        l_freq=7, h_freq=30, fir_design="firwin", skip_by_annotation="edge"
    )
    raw_pp.plot(title="Preprocessed signal (7-30 Hz bandpass)")
    plt.show()
