import argparse

from src.modules.inference import run_inference
from src.models.generator import Generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference to convert voice.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input audio file (.wav or .mp3)."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of saved Generator model (from Jar)."
    )
    parser.add_argument("--output", type=str, required=True, help="Path to save output audio.")
    args = parser.parse_args()

    generator = Generator.load(args.model)

    run_inference(
        input_path=args.input,
        generator=generator,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
