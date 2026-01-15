# src/models/pretrained.py
"""Pretrained model management for RVC."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import urllib.request
import torch
import torch.nn as nn

# Default pretrained model URLs from RVC project (HuggingFace)
PRETRAINED_URLS = {
    "v2_48k_g": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth",
    "v2_48k_d": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth",
    "v2_40k_g": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
    "v2_40k_d": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth",
    "v2_32k_g": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G32k.pth",
    "v2_32k_d": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D32k.pth",
}

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rvc_pretrained"


def get_cache_dir() -> Path:
    """Get the cache directory for pretrained models."""
    cache_dir = Path(os.environ.get("RVC_CACHE_DIR", DEFAULT_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_pretrained(
    model_key: str,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download a pretrained model from HuggingFace.

    Args:
        model_key: Key from PRETRAINED_URLS (e.g., "v2_48k_g")
        cache_dir: Directory to cache downloads
        force: Force re-download even if cached

    Returns:
        Path to downloaded model file
    """
    if model_key not in PRETRAINED_URLS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(PRETRAINED_URLS.keys())}")

    url = PRETRAINED_URLS[model_key]
    cache_dir = cache_dir or get_cache_dir()
    filename = f"{model_key}.pth"
    filepath = cache_dir / filename

    if filepath.exists() and not force:
        print(f"Using cached pretrained model: {filepath}")
        return filepath

    print(f"Downloading pretrained model: {model_key}")
    print(f"  URL: {url}")
    print(f"  Destination: {filepath}")

    # Download with progress
    def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            print(f"\r  Progress: {percent}% ({downloaded // 1024 // 1024}MB)", end="", flush=True)

    urllib.request.urlretrieve(url, filepath, reporthook=_progress_hook)
    print("\n  Download complete!")

    return filepath


def load_pretrained_state_dict(
    model_key: str,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load pretrained state dict, downloading if necessary.

    Args:
        model_key: Key from PRETRAINED_URLS
        cache_dir: Directory to cache downloads

    Returns:
        State dictionary
    """
    filepath = download_pretrained(model_key, cache_dir)
    return torch.load(filepath, map_location="cpu", weights_only=False)


def _remap_generator_keys(sd: dict[str, Any]) -> dict[str, Any]:
    """
    Remap RVC pretrained generator keys to our architecture.

    The reference RVC uses slightly different naming conventions.
    """
    remapped: dict[str, Any] = {}

    for key, value in sd.items():
        new_key = key

        # The reference uses "dec." prefix for decoder/generator
        # Our architecture uses direct naming under synthesizer.dec
        if key.startswith("dec."):
            new_key = key  # Keep as-is, we use same structure

        # enc_p (text encoder) mappings
        if key.startswith("enc_p."):
            new_key = key

        # enc_q (posterior encoder) mappings
        if key.startswith("enc_q."):
            new_key = key

        # flow mappings
        if key.startswith("flow."):
            new_key = key

        # emb_g (speaker embedding) - skip if single speaker
        if key.startswith("emb_g."):
            continue  # Skip speaker embeddings for single-speaker training

        remapped[new_key] = value

    return remapped


def _remap_discriminator_keys(sd: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Remap RVC pretrained discriminator keys.

    Returns:
        Tuple of (mpd_state_dict, msd_state_dict)
    """
    mpd_sd: dict[str, Any] = {}
    msd_sd: dict[str, Any] = {}

    for key, value in sd.items():
        # MPD keys
        if key.startswith("mpd.") or key.startswith("discriminators."):
            # Remove prefix and remap
            if key.startswith("mpd."):
                new_key = key[4:]  # Remove "mpd."
            else:
                new_key = key
            mpd_sd[new_key] = value

        # MSD keys
        elif key.startswith("msd."):
            new_key = key[4:]  # Remove "msd."
            msd_sd[new_key] = value

    return mpd_sd, msd_sd


def load_pretrained_into_module(
    module: nn.Module,
    state_dict: dict[str, Any],
    strict: bool = False,
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Load pretrained weights into a module with shape matching.

    Args:
        module: Target module
        state_dict: Source state dict
        strict: If True, raise error on mismatches
        verbose: Print loading info

    Returns:
        Tuple of (loaded_keys, skipped_keys)
    """
    target_sd = module.state_dict()
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []

    filtered_sd: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key in target_sd:
            if hasattr(value, "shape") and value.shape == target_sd[key].shape:
                filtered_sd[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {target_sd[key].shape})")
        else:
            skipped_keys.append(f"{key} (not in target)")

    if strict and skipped_keys:
        raise ValueError(f"Strict loading failed. Skipped keys: {skipped_keys}")

    module.load_state_dict(filtered_sd, strict=False)

    if verbose:
        print(f"  Loaded {len(loaded_keys)} / {len(state_dict)} parameters")
        if skipped_keys and len(skipped_keys) <= 10:
            print(f"  Skipped: {skipped_keys}")
        elif skipped_keys:
            print(f"  Skipped {len(skipped_keys)} parameters (shape mismatches or missing)")

    return loaded_keys, skipped_keys


class PretrainedManager:
    """
    Manager for downloading and loading pretrained RVC models.

    Usage:
        manager = PretrainedManager(sample_rate=48000)
        manager.load_into_generator(generator)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        version: str = "v2",
        cache_dir: Path | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.version = version
        self.cache_dir = cache_dir or get_cache_dir()

        # Determine model keys based on sample rate
        sr_key = f"{sample_rate // 1000}k"
        self.g_key = f"{version}_{sr_key}_g"
        self.d_key = f"{version}_{sr_key}_d"

        if self.g_key not in PRETRAINED_URLS:
            available_srs = [k.split("_")[1] for k in PRETRAINED_URLS if k.startswith(version)]
            raise ValueError(
                f"No pretrained model for {sample_rate}Hz. "
                f"Available sample rates: {available_srs}"
            )

    def download_all(self, force: bool = False) -> tuple[Path, Path]:
        """Download both generator and discriminator pretrained models."""
        g_path = download_pretrained(self.g_key, self.cache_dir, force)
        d_path = download_pretrained(self.d_key, self.cache_dir, force)
        return g_path, d_path

    def load_generator_state_dict(self) -> dict[str, Any]:
        """Load and remap generator state dict."""
        sd = load_pretrained_state_dict(self.g_key, self.cache_dir)
        return _remap_generator_keys(sd)

    def load_discriminator_state_dicts(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load and remap discriminator state dicts."""
        sd = load_pretrained_state_dict(self.d_key, self.cache_dir)
        return _remap_discriminator_keys(sd)

    def load_into_synthesizer(
        self,
        synthesizer: nn.Module,
        verbose: bool = True,
    ) -> None:
        """Load pretrained weights into synthesizer."""
        if verbose:
            print(f"Loading pretrained generator weights ({self.g_key})...")

        sd = self.load_generator_state_dict()
        load_pretrained_into_module(synthesizer, sd, strict=False, verbose=verbose)

    def load_into_discriminators(
        self,
        mpd: nn.Module,
        msd: nn.Module,
        verbose: bool = True,
    ) -> None:
        """Load pretrained weights into discriminators."""
        if verbose:
            print(f"Loading pretrained discriminator weights ({self.d_key})...")

        mpd_sd, msd_sd = self.load_discriminator_state_dicts()

        if verbose:
            print("  MPD:")
        load_pretrained_into_module(mpd, mpd_sd, strict=False, verbose=verbose)

        if verbose:
            print("  MSD:")
        load_pretrained_into_module(msd, msd_sd, strict=False, verbose=verbose)


def download_all_pretrained(
    sample_rate: int = 48000,
    version: str = "v2",
) -> None:
    """
    Convenience function to download all pretrained models.

    Args:
        sample_rate: Target sample rate (32000, 40000, or 48000)
        version: Model version ("v2")
    """
    manager = PretrainedManager(sample_rate=sample_rate, version=version)
    manager.download_all()
    print("All pretrained models downloaded!")
