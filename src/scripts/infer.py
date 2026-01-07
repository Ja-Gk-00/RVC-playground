# src/scripts/infer.py
import argparse

from src.modules.inference import run_inference
from src.models.generator import Generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference to convert voice.")
    parser.add_argument("--input", required=True, help="Path to input audio file (.wav/.flac/.mp3).")
    parser.add_argument("--model", required=True, help="Name of saved Generator model (from Jar).")
    parser.add_argument("--output", required=True, help="Path to save output audio.")
    parser.add_argument("--hubert_path", "--hubert-path", dest="hubert_path", default=None)
    parser.add_argument("--rmvpe_path", "--rmvpe-path", dest="rmvpe_path", default=None)
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    args = parser.parse_args()

    generator = Generator.load(args.model)

    run_inference(
        input_path=args.input,
        generator=generator,
        output_path=args.output,
        hubert_path=args.hubert_path,
        rmvpe_path=args.rmvpe_path,
        normalize_output=bool(args.normalize),
    )


if __name__ == "__main__":
    main()
