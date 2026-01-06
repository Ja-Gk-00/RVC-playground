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
        "--target_sr",
        str(args.target_sr),
        "--learning_rate",
        str(args.learning_rate),
    ]
    if args.content_dim is not None:
        cmd += ["--content_dim", str(args.content_dim)]
    if args.use_pitch:
        cmd.append("--use_pitch")
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
    ]
    if args.hubert_path:
        cmd += ["--hubert_path", args.hubert_path]
    if args.rmvpe_path:
        cmd += ["--rmvpe_path", args.rmvpe_path]
    if args.no_normalize:
        cmd.append("--no_normalize")
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="RVC CLI")
    subparsers = parser.add_subparsers(dest="command")

    pp = subparsers.add_parser("preprocess")
    pp.add_argument("--data_dir", "--data-dir", dest="data_dir", required=True)
    pp.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    pp.add_argument("--hubert_path", "--hubert-path", dest="hubert_path", default=None)
    pp.add_argument("--rmvpe_path", "--rmvpe-path", dest="rmvpe_path", default=None)
    pp.add_argument("--target_sr", "--target-sr", dest="target_sr", type=int, default=48000)
    pp.add_argument("--quiet", action="store_true")
    pp.set_defaults(func=run_preprocess)

    tr = subparsers.add_parser("train")
    tr.add_argument("--feature_dir", "--feature-dir", dest="feature_dir", required=True)
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=4)
    tr.add_argument("--model_name", "--model-name", dest="model_name", default="my_voice_rvc")
    tr.add_argument("--content_dim", "--content-dim", dest="content_dim", type=int, default=None)
    tr.add_argument("--use_pitch", "--use-pitch", dest="use_pitch", action="store_true")
    tr.add_argument("--target_sr", "--target-sr", dest="target_sr", type=int, default=48000)
    tr.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=1e-4)
    tr.set_defaults(func=run_train)

    inf = subparsers.add_parser("infer")
    inf.add_argument("--input", required=True)
    inf.add_argument("--model", required=True)
    inf.add_argument("--output", required=True)
    inf.add_argument("--hubert_path", "--hubert-path", dest="hubert_path", default=None)
    inf.add_argument("--rmvpe_path", "--rmvpe-path", dest="rmvpe_path", default=None)
    inf.add_argument("--no_normalize", dest="no_normalize", action="store_true")
    inf.set_defaults(func=run_infer)

    args = parser.parse_args()
    if hasattr(args, "func"):
        raise SystemExit(args.func(args))
    parser.print_help()


if __name__ == "__main__":
    main()
