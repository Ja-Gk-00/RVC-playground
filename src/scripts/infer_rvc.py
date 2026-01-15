#!/usr/bin/env python3
"""
RVC Inference Script

Convert audio using a trained RVC model.

Usage:
    # Basic voice conversion (using pretrained model)
    python -m src.scripts.infer_rvc --input input.wav --model model.pth --output output.wav

    # With pitch shift (positive = higher, negative = lower)
    python -m src.scripts.infer_rvc --input input.wav --model model.pth --output output.wav --pitch 4

    # Using FAISS index for better quality
    python -m src.scripts.infer_rvc --input input.wav --model model.pth --output output.wav --index model.index

Example workflow:
    1. Fine-tune a model on your voice:
       python -m src.scripts.finetune --data_dir ./my_voice --pretrained pretrained.pth --output my_model.pth

    2. Convert audio to your voice:
       python -m src.scripts.infer_rvc --input song.wav --model my_model.pth --output converted.wav
"""
import argparse
import os
import sys
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(
        description="RVC Voice Conversion Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert voice using model
  python -m src.scripts.infer_rvc -i input.wav -m model.pth -o output.wav

  # Shift pitch up by 4 semitones
  python -m src.scripts.infer_rvc -i input.wav -m model.pth -o output.wav -p 4

  # Use FAISS index for better timbre matching
  python -m src.scripts.infer_rvc -i input.wav -m model.pth -o output.wav --index model.index
        """
    )

    # Required arguments
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input audio file (WAV, MP3, FLAC)")
    parser.add_argument("-m", "--model", required=True,
                        help="Path to RVC model file (.pth)")
    parser.add_argument("-o", "--output", required=True,
                        help="Path for output audio (WAV)")

    # Optional arguments
    parser.add_argument("-p", "--pitch", type=int, default=0,
                        help="Pitch shift in semitones (default: 0). "
                             "Use +12 to shift up one octave, -12 to shift down.")
    parser.add_argument("--sid", type=int, default=0,
                        help="Speaker ID for multi-speaker models (default: 0)")
    parser.add_argument("--index", default=None,
                        help="Path to FAISS index file for improved quality")
    parser.add_argument("--index-rate", type=float, default=0.5,
                        help="Index mixing rate 0-1 (default: 0.5). "
                             "Higher = more timbre matching, lower = more original")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--half", action="store_true",
                        help="Use half precision (FP16) for faster inference on GPU")
    parser.add_argument("--chunk-seconds", type=float, default=10.0,
                        help="Process in chunks of N seconds to save memory (default: 10). "
                             "Use smaller values (5-8) if you get OOM errors.")
    parser.add_argument("--protect", type=float, default=0.33,
                        help="Protect voiceless consonants (0-0.5, default: 0.33). "
                             "Higher values preserve more consonant clarity.")

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Validate model file
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = Path(args.output).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    print("=" * 50)
    print("RVC Voice Conversion")
    print("=" * 50)
    print(f"Input:  {args.input}")
    print(f"Model:  {args.model}")
    print(f"Output: {args.output}")
    if args.pitch != 0:
        print(f"Pitch:  {args.pitch:+d} semitones")
    if args.index:
        print(f"Index:  {args.index} (rate: {args.index_rate})")
    print()

    # Import and run
    from src.modules.inference_rvc import run_inference_rvc

    run_inference_rvc(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        speaker_id=args.sid,
        f0_up_key=args.pitch,
        index_path=args.index,
        index_rate=args.index_rate,
        device=args.device,
        chunk_seconds=args.chunk_seconds,
        protect=args.protect,
    )

    print()
    print("=" * 50)
    print("Done! Output saved to:", args.output)
    print("=" * 50)


if __name__ == "__main__":
    main()