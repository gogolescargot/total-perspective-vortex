from argparse import ArgumentParser
import visualize
import train
import predict


def parse_args():
    p = ArgumentParser(description="total-perspective-vortex")

    p.add_argument(
        "mode",
        choices=["visualize", "train", "predict"],
        help="Mode to run (positional, no dashes)",
    )
    p.add_argument(
        "--subjects", "-s", type=int, nargs="+", default=list(range(1, 110))
    )
    p.add_argument(
        "--path",
        "-p",
        type=str,
        default=None,
        help="Path to MNE/eegbci data",
    )
    p.add_argument("--runs", "-r", type=int, nargs="+")
    p.add_argument("--experiment", "-e", type=int, default=None)
    p.add_argument("--out", "-o", type=str, default="model.joblib")

    args = p.parse_args()

    if args.subjects is not None:
        args.subjects = list(dict.fromkeys(args.subjects))

    if args.runs is not None:
        args.runs = list(dict.fromkeys(args.runs))

    return args


def main():
    try:
        args = parse_args()
        if args.mode == "visualize":
            visualize.visualize(args.subjects, args.runs, args.path)
        elif args.mode == "train":
            train.train(
                args.subjects,
                args.runs,
                args.experiment,
                args.out,
                args.path,
            )
        elif args.mode == "predict":
            predict.predict(
                args.subjects,
                args.runs,
                args.experiment,
                args.out,
                args.path,
            )

    except KeyboardInterrupt:
        print("Process interrupted by user, exiting.")
    except ValueError as ve:
        print(f"Value error: {ve}")
    except RuntimeError as re:
        print(f"Runtime error: {re}")
    except FileNotFoundError as fnfe:
        print(f"File not found: {fnfe}")
    except Exception as ex:
        print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()
