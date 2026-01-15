# src/scripts/cli.py
import argparse
import subprocess
import sys


def run_preprocess(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.preprocess",
        "--data_dir",
        args.data_dir,
        "--out_dir",
        args.out_dir,
        "--target_sr",
        str(args.target_sr),
        "--segment_duration",
        str(args.segment_duration),
        "--segment_overlap",
        str(args.segment_overlap),
    ]
    if args.hubert_path:
        cmd += ["--hubert_path", args.hubert_path]
    if args.rmvpe_path:
        cmd += ["--rmvpe_path", args.rmvpe_path]
    if args.quiet:
        cmd.append("--quiet")
    return subprocess.run(cmd, check=False).returncode


def run_train(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.train",
        "--feature_dir",
        args.feature_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--model_name",
        args.model_name,
        "--content_dim",
        str(args.content_dim),
        "--target_sr",
        str(args.target_sr),
        "--learning_rate",
        str(args.learning_rate)
    ]
    if args.use_pitch:
        cmd.append("--use_pitch")
    if args.fp16:
        cmd.append("--fp16")
    if args.device:
        cmd += ["--device", args.device]
    return subprocess.run(cmd, check=False).returncode


def run_infer(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.infer",
        "--input",
        args.input,
        "--model",
        args.model,
        "--output",
        args.output,
        "--target_sr",
        str(args.target_sr),
        "--index_rate",
        str(args.index_rate),
        "--k",
        str(args.k),
    ]
    if args.hubert_path:
        cmd += ["--hubert_path", args.hubert_path]
    if args.rmvpe_path:
        cmd += ["--rmvpe_path", args.rmvpe_path]
    if args.use_pitch:
        cmd.append("--use_pitch")
    if args.index:
        cmd += ["--index", args.index]
    if args.index_data:
        cmd += ["--index_data", args.index_data]
    if args.device:
        cmd += ["--device", args.device]
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="RVC CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # preprocess
    pp = subparsers.add_parser("preprocess")
    pp.add_argument("--data_dir", "--data-dir", dest="data_dir", required=True)
    pp.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    pp.add_argument("--hubert_path", default=None)
    pp.add_argument("--rmvpe_path", default=None)
    pp.add_argument("--target_sr", type=int, default=48000)
    pp.add_argument("--segment_duration", type=float, default=4.0)
    pp.add_argument("--segment_overlap", type=float, default=0.3)
    pp.add_argument("--quiet", action="store_true")
    pp.set_defaults(func=run_preprocess)

    # train
    tr = subparsers.add_parser("train")
    tr.add_argument("--feature_dir", "--feature-dir", dest="feature_dir", required=True)
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--batch_size", type=int, default=2)
    tr.add_argument("--model_name", default="my_voice_rvc")
    tr.add_argument("--content_dim", type=int, default=768)
    tr.add_argument("--use_pitch", action="store_true")
    tr.add_argument("--target_sr", type=int, default=48000)
    tr.add_argument("--learning_rate", type=float, default=1e-4)
    tr.add_argument("--save_every", type=int, default=1)
    tr.add_argument("--fp16", action="store_true")
    tr.add_argument("--device", default=None)  # e.g. "cuda" or "cpu"
    tr.set_defaults(func=run_train)

    # infer
    inf = subparsers.add_parser("infer")
    inf.add_argument("--input", required=True)
    inf.add_argument("--model", required=True)   # Jar object name, not a path
    inf.add_argument("--output", required=True)
    inf.add_argument("--hubert_path", default=None)
    inf.add_argument("--rmvpe_path", default=None)
    inf.add_argument("--use_pitch", action="store_true")
    inf.add_argument("--target_sr", type=int, default=48000)
    inf.add_argument("--index", default=None)        # path to faiss_index.index
    inf.add_argument("--index_data", default=None)   # path to content_vectors.npy
    inf.add_argument("--index_rate", type=float, default=0.5)
    inf.add_argument("--k", type=int, default=8)
    inf.add_argument("--device", default=None)
    inf.set_defaults(func=run_infer)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
