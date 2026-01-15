# src/scripts/infer.py
import argparse
from src.modules.inference import run_inference


def main() -> None:
    p = argparse.ArgumentParser(description="RVC inference (Jar model name + optional FAISS retrieval).")
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True)   # Jar object name
    p.add_argument("--output", required=True)

    p.add_argument("--hubert_path", default=None)
    p.add_argument("--rmvpe_path", default=None)
    p.add_argument("--use_pitch", action="store_true")
    p.add_argument("--target_sr", type=int, default=48000)
    p.add_argument("--device", default=None)

    p.add_argument("--index", default=None)
    p.add_argument("--index_data", default=None)
    p.add_argument("--index_rate", type=float, default=0.5)
    p.add_argument("--k", type=int, default=8)

    args = p.parse_args()

    run_inference(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        hubert_path=args.hubert_path,
        rmvpe_path=args.rmvpe_path,
        use_pitch=args.use_pitch,
        target_sr=args.target_sr,
        device=args.device,
        index_path=args.index,
        index_data_path=args.index_data,
        index_rate=args.index_rate,
        k=args.k,
    )


if __name__ == "__main__":
    main()
