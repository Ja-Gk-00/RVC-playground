#!/usr/bin/env python3
"""
Download required pretrained models for RVC-Playground.

This script downloads:
- HuBERT Base model (required for content feature extraction)
- RMVPE model (required for pitch extraction)
- Pretrained RVC weights (optional, for transfer learning)

Usage:
    python -m src.scripts.download_models           # Download required models only
    python -m src.scripts.download_models --all     # Download all models including pretrained RVC
    python -m src.scripts.download_models --list    # List available models
"""

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Model storage paths
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent
SAVED_MODELS_DIR = SRC_DIR / "saved_models"
HUBERT_DIR = SAVED_MODELS_DIR / "hubert"
RMVPE_DIR = SAVED_MODELS_DIR / "rmvpe"
PRETRAINED_RVC_DIR = SAVED_MODELS_DIR / "pretrained_rvc"


@dataclass
class ModelInfo:
    """Information about a model to download."""

    name: str
    url: str
    path: Path
    description: str
    required: bool = True
    sha256: str | None = None  # Optional checksum for verification


# Model definitions
MODELS: dict[str, ModelInfo] = {
    "hubert": ModelInfo(
        name="HuBERT Base",
        url="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        path=HUBERT_DIR / "hubert_base.pt",
        description="Content feature extraction model (required)",
        required=True,
    ),
    "rmvpe": ModelInfo(
        name="RMVPE",
        url="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        path=RMVPE_DIR / "rmvpe.pt",
        description="Pitch extraction model (required)",
        required=True,
    ),
    "pretrained_g_48k": ModelInfo(
        name="Pretrained Generator (48kHz)",
        url="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth",
        path=PRETRAINED_RVC_DIR / "f0G48k.pth",
        description="Pretrained generator for transfer learning (48kHz)",
        required=False,
    ),
    "pretrained_d_48k": ModelInfo(
        name="Pretrained Discriminator (48kHz)",
        url="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth",
        path=PRETRAINED_RVC_DIR / "f0D48k.pth",
        description="Pretrained discriminator for transfer learning (48kHz)",
        required=False,
    ),
    "pretrained_g_40k": ModelInfo(
        name="Pretrained Generator (40kHz)",
        url="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
        path=PRETRAINED_RVC_DIR / "f0G40k.pth",
        description="Pretrained generator for transfer learning (40kHz)",
        required=False,
    ),
    "pretrained_d_40k": ModelInfo(
        name="Pretrained Discriminator (40kHz)",
        url="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth",
        path=PRETRAINED_RVC_DIR / "f0D40k.pth",
        description="Pretrained discriminator for transfer learning (40kHz)",
        required=False,
    ),
    "pretrained_g_32k": ModelInfo(
        name="Pretrained Generator (32kHz)",
        url="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G32k.pth",
        path=PRETRAINED_RVC_DIR / "f0G32k.pth",
        description="Pretrained generator for transfer learning (32kHz)",
        required=False,
    ),
    "pretrained_d_32k": ModelInfo(
        name="Pretrained Discriminator (32kHz)",
        url="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D32k.pth",
        path=PRETRAINED_RVC_DIR / "f0D32k.pth",
        description="Pretrained discriminator for transfer learning (32kHz)",
        required=False,
    ),
}


class DownloadProgressBar:
    """Progress bar for urllib downloads."""

    def __init__(self, filename: str):
        self.filename = filename
        self.downloaded = 0
        self.total = 0
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            self.total = total_size
            self.downloaded = block_num * block_size
            percent = min(100, (self.downloaded * 100) // total_size)

            if percent != self.last_percent:
                self.last_percent = percent
                downloaded_mb = self.downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                bar_length = 40
                filled = int(bar_length * percent // 100)
                bar = "=" * filled + "-" * (bar_length - filled)
                print(
                    f"\r  [{bar}] {percent:3d}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)",
                    end="",
                    flush=True,
                )
        else:
            self.downloaded += block_size
            downloaded_mb = self.downloaded / (1024 * 1024)
            print(f"\r  Downloaded: {downloaded_mb:.1f} MB", end="", flush=True)


def verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    """Verify file checksum."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_sha256


def download_model(model: ModelInfo, force: bool = False) -> bool:
    """
    Download a single model.

    Args:
        model: ModelInfo object with download details
        force: If True, download even if file exists

    Returns:
        True if successful, False otherwise
    """
    # Check if already exists
    if model.path.exists() and not force:
        size_mb = model.path.stat().st_size / (1024 * 1024)
        print(f"  Already exists: {model.path} ({size_mb:.1f} MB)")
        return True

    # Create directory
    model.path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading from: {model.url}")
    print(f"  Saving to: {model.path}")

    try:
        progress = DownloadProgressBar(model.path.name)
        urllib.request.urlretrieve(model.url, model.path, reporthook=progress)
        print()  # New line after progress bar

        # Verify download
        if model.path.exists():
            size_mb = model.path.stat().st_size / (1024 * 1024)
            print(f"  Downloaded successfully ({size_mb:.1f} MB)")

            # Verify checksum if available
            if model.sha256:
                print("  Verifying checksum...", end=" ")
                if verify_checksum(model.path, model.sha256):
                    print("OK")
                else:
                    print("FAILED")
                    print("  Warning: Checksum mismatch, file may be corrupted")
                    return False
            return True
        else:
            print("  Error: Download failed, file not found")
            return False

    except urllib.error.URLError as e:
        print(f"\n  Error: Failed to download - {e}")
        return False
    except Exception as e:
        print(f"\n  Error: {e}")
        # Clean up partial download
        if model.path.exists():
            model.path.unlink()
        return False


def list_models() -> None:
    """List all available models and their status."""
    print("\nAvailable models:")
    print("-" * 80)

    for key, model in MODELS.items():
        status = "INSTALLED" if model.path.exists() else "NOT INSTALLED"
        required = "required" if model.required else "optional"
        size = ""
        if model.path.exists():
            size_mb = model.path.stat().st_size / (1024 * 1024)
            size = f" ({size_mb:.1f} MB)"

        print(f"\n  {key}:")
        print(f"    Name: {model.name}")
        print(f"    Description: {model.description}")
        print(f"    Status: {status}{size}")
        print(f"    Type: {required}")
        print(f"    Path: {model.path}")


def download_required_models(force: bool = False) -> bool:
    """Download only required models."""
    print("\n" + "=" * 60)
    print("Downloading required models for RVC-Playground")
    print("=" * 60)

    success = True
    for key, model in MODELS.items():
        if model.required:
            print(f"\n[{model.name}]")
            if not download_model(model, force):
                success = False

    return success


def download_all_models(force: bool = False) -> bool:
    """Download all models including optional pretrained weights."""
    print("\n" + "=" * 60)
    print("Downloading all models for RVC-Playground")
    print("=" * 60)

    success = True
    for key, model in MODELS.items():
        print(f"\n[{model.name}]")
        if not download_model(model, force):
            success = False

    return success


def download_specific_models(model_keys: list[str], force: bool = False) -> bool:
    """Download specific models by key."""
    print("\n" + "=" * 60)
    print("Downloading selected models")
    print("=" * 60)

    success = True
    for key in model_keys:
        if key not in MODELS:
            print(f"\n[{key}] - Unknown model, skipping")
            continue

        model = MODELS[key]
        print(f"\n[{model.name}]")
        if not download_model(model, force):
            success = False

    return success


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download pretrained models for RVC-Playground",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.scripts.download_models              # Download required models
  python -m src.scripts.download_models --all        # Download all models
  python -m src.scripts.download_models --list       # List available models
  python -m src.scripts.download_models hubert rmvpe # Download specific models
  python -m src.scripts.download_models --force      # Re-download even if exists
        """,
    )

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Download all models including optional pretrained RVC weights",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available models and their status",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if models already exist",
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Specific models to download (e.g., hubert rmvpe pretrained_g_48k)",
    )

    args = parser.parse_args()

    # List models
    if args.list:
        list_models()
        return 0

    # Download specific models
    if args.models:
        success = download_specific_models(args.models, force=args.force)
    # Download all models
    elif args.all:
        success = download_all_models(force=args.force)
    # Download required models only
    else:
        success = download_required_models(force=args.force)

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("All downloads completed successfully!")
        print("\nModel locations:")
        print(f"  HuBERT:  {HUBERT_DIR}")
        print(f"  RMVPE:   {RMVPE_DIR}")
        if args.all or any("pretrained" in m for m in args.models if args.models):
            print(f"  RVC:     {PRETRAINED_RVC_DIR}")
    else:
        print("Some downloads failed. Please check the errors above and try again.")
        return 1

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
