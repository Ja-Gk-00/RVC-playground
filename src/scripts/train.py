import argparse

from src.modules.training import train_generator_from_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RVC generator model.")
    parser.add_argument("--feature_dir", required=True, help="Directory with preprocessed features.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--model_name", default="trained_generator", help="Name for saved model.")
    parser.add_argument("--content_dim", type=int, default=256, help="Content feature dimension.")
    parser.add_argument("--use_pitch", action="store_true", help="Use pitch conditioning.")
    parser.add_argument("--target_sr", type=int, default=48000, help="Target sample rate.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    args = parser.parse_args()

    train_generator_from_features(
        feature_dir=args.feature_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        content_dim=args.content_dim,
        use_pitch=args.use_pitch,
        target_sr=args.target_sr,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
