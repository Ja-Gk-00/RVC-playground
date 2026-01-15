# src/scripts/preprocess.py
import argparse
from src.modules.preprocessing import preprocess_for_training


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess audio -> units/f0/audio + FAISS index.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hubert_path", default=None)
    p.add_argument("--rmvpe_path", default=None)
    p.add_argument("--target_sr", type=int, default=48000)
    p.add_argument("--segment_duration", type=float, default=4.0)
    p.add_argument("--segment_overlap", type=float, default=0.3)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    preprocess_for_training(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        hubert_path=args.hubert_path,
        rmvpe_path=args.rmvpe_path,
        target_sr=args.target_sr,
        segment_duration=args.segment_duration,
        segment_overlap=args.segment_overlap,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
