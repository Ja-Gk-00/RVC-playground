# src/scripts/train.py
import argparse

from src.modules.training import train_generator_from_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RVC generator model.")
    parser.add_argument("--feature_dir", "--feature-dir", dest="feature_dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--model_name", "--model-name", dest="model_name", default="my_voice_rvc")
    parser.add_argument("--content_dim", "--content-dim", dest="content_dim", type=int, default=None)
    parser.add_argument("--use_pitch", "--use-pitch", dest="use_pitch", action="store_true")
    parser.add_argument("--target_sr", "--target-sr", dest="target_sr", type=int, default=48000)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=1e-4)
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
