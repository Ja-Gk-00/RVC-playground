import argparse
from src.modules.training import train_generator_from_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the RVC generator model (GAN: G/D).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch (not recommended - slow)
  python -m src.scripts.train --feature_dir data/features --epochs 500

  # Fine-tune from official RVC pretrained weights (RECOMMENDED)
  python -m src.scripts.train --feature_dir data/features --pretrained_rvc --epochs 200

  # Fine-tune with custom pretrained weights
  python -m src.scripts.train --feature_dir data/features --pretrained_g path/to/G.pth
        """,
    )
    parser.add_argument("--feature_dir", required=True, help="Directory with preprocessed features.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4 for 4-8GB GPU)")
    parser.add_argument("--model_name", default="my_voice_rvc", help="Jar name to save into.")
    parser.add_argument("--content_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=192, help="Latent dimension (default: 192)")
    parser.add_argument("--target_sr", type=int, default=48000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # Pretrained model options
    pretrained_group = parser.add_argument_group("Pretrained Models")
    pretrained_group.add_argument(
        "--pretrained_rvc",
        action="store_true",
        help="Load official RVC pretrained weights from HuggingFace (RECOMMENDED)",
    )
    pretrained_group.add_argument("--pretrained", default=None, help="Path to custom pretrained model")
    pretrained_group.add_argument("--pretrained_g", default=None, help="Path to custom pretrained generator")
    pretrained_group.add_argument("--pretrained_d", default=None, help="Path to custom pretrained discriminator")

    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 training (default: True)")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 training")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--stats_csv", default=None, help="Path to CSV file for logging training statistics")
    parser.add_argument(
        "--metrics_every",
        type=int,
        default=None,
        help="Compute expensive metrics (F0 corr, speaker sim) every N epochs (default: 5)",
    )

    args = parser.parse_args()

    # Handle fp16 flag
    fp16 = args.fp16 and not args.no_fp16

    train_generator_from_features(
        feature_dir=args.feature_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        content_dim=args.content_dim,
        hidden_dim=args.hidden_dim,
        target_sr=args.target_sr,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
        pretrained_rvc=args.pretrained_rvc,
        pretrained=args.pretrained,
        pretrained_g=args.pretrained_g,
        pretrained_d=args.pretrained_d,
        fp16=fp16,
        device=args.device,
        stats_csv=args.stats_csv,
        expensive_metrics_every=args.metrics_every,
    )


if __name__ == "__main__":
    main()
