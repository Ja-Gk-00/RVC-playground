import argparse

from src.modules.preprocessing import preprocess_for_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio data for RVC.")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory with raw audio files (wav/flac)."
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Output directory for features and index."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output."
    )
    args = parser.parse_args()

    preprocess_for_training(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
