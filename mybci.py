from argparse import ArgumentParser
import visualize
import train
import predict
import signal


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
    p.add_argument("--runs", "-r", type=int, nargs="+")
    p.add_argument("--experiment", "-e", type=int, default=None)
    p.add_argument("--out", "-o", type=str, default="model.joblib")
    return p.parse_args()


def main():
    # try:
    signal.signal(
        signal.SIGINT,
        lambda *_: (
            print("\033[2Dtotal-perspective-vortex: CTRL+C sent by user."),
            exit(1),
        ),
    )

    args = parse_args()
    if args.mode == "visualize":
        visualize.visualize(args.subjects, args.runs)
    elif args.mode == "train":
        train.train(args.subjects, args.runs, args.experiment, args.out)
    elif args.mode == "predict":
        predict.predict(args.subjects, args.runs, args.experiment, args.out)


# except Exception as ex:
# print(f"Unexpected error occured : {ex}")


if __name__ == "__main__":
    main()
