# src/scripts/preprocess.py
import argparse

from src.modules.preprocessing import preprocess_for_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio data for RVC.")
    parser.add_argument("--data_dir", "--data-dir", dest="data_dir", required=True)
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    parser.add_argument("--hubert_path", "--hubert-path", dest="hubert_path", default=None)
    parser.add_argument("--rmvpe_path", "--rmvpe-path", dest="rmvpe_path", default=None)
    parser.add_argument("--target_sr", "--target-sr", dest="target_sr", type=int, default=48000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    preprocess_for_training(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        hubert_path=args.hubert_path,
        rmvpe_path=args.rmvpe_path,
        target_sr=args.target_sr,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
