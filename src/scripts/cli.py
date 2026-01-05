import argparse
import subprocess


def run_preprocess(args):
    cmd = [
        "python",
        "-m",
        "src.scripts.preprocess",
        "--data_dir",
        args.data_dir,
        "--out_dir",
        args.out_dir,
    ]
    if args.hubert_path:
        cmd += ["--hubert_path", args.hubert_path]
    if args.rmvpe_path:
        cmd += ["--rmvpe_path", args.rmvpe_path]
    subprocess.run(cmd)


def run_train(args):
    cmd = [
        "python",
        "-m",
        "src.scripts.train",
        "--feature_dir",
        args.feature_dir,
        "--data_dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--out_model",
        args.out_model,
        "--content_dim",
        str(args.content_dim),
        "--target_sr",
        str(args.target_sr),
    ]
    if args.use_pitch:
        cmd.append("--use_pitch")
    subprocess.run(cmd)


def run_infer(args):
    cmd = [
        "python",
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
    if args.use_pitch:
        cmd.append("--use_pitch")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="RVC CLI")
    subparsers = parser.add_subparsers(dest="command")

    pp = subparsers.add_parser("preprocess")
    pp.add_argument("--data_dir", required=True)
    pp.add_argument("--out_dir", required=True)
    pp.add_argument("--hubert_path")
    pp.add_argument("--rmvpe_path")
    pp.set_defaults(func=run_preprocess)

    tr = subparsers.add_parser("train")
    tr.add_argument("--feature_dir", required=True)
    tr.add_argument("--data_dir", required=True)
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch_size", type=int, default=1)
    tr.add_argument("--out_model", default="generator.pth")
    tr.add_argument("--content_dim", type=int, default=256)
    tr.add_argument("--use_pitch", action="store_true")
    tr.add_argument("--target_sr", type=int, default=48000)
    tr.set_defaults(func=run_train)

    inf = subparsers.add_parser("infer")
    inf.add_argument("--input", required=True)
    inf.add_argument("--model", required=True)
    inf.add_argument("--output", required=True)
    inf.add_argument("--hubert_path")
    inf.add_argument("--rmvpe_path")
    inf.add_argument("--use_pitch", action="store_true")
    inf.set_defaults(func=run_infer)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
