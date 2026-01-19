import mne
from mne.datasets import eegbci
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser(
        prog="Data",
        description="Preprocessing, parsing and formatting.",
    )

    parser.add_argument(
        "--subject",
        type=int,
        help="Subject number to load.",
        default=1,
    )

    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        help="List of runs to load.",
        default=[6, 10],
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    subject = 1
    runs = args.runs

    paths = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(p, preload=True) for p in paths]
    raw = mne.concatenate_raws(raws)
    raw.plot(title="Raw signal")
    plt.show()

    bands = {"alpha": (8, 13), "beta": (13, 30)}

    raw_alpha = raw.copy().filter(
        l_freq=bands["alpha"][0], h_freq=bands["alpha"][1]
    )
    raw_alpha.plot(title="Alpha (8-13 Hz)")
    plt.show()

    raw_beta = raw.copy().filter(
        l_freq=bands["beta"][0], h_freq=bands["beta"][1]
    )
    raw_beta.plot(title="Beta (13-30 Hz)")
    plt.show()


if __name__ == "__main__":
    main()
