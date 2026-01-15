from __future__ import annotations

import inspect
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy import signal
from tqdm import tqdm

from src.data_models.data_models import AudioSegment, SegmentedAudio, UnprocessedTrainingData


class Preprocessor:
    AUDIO_EXTENSIONS = {".wav", ".flac"}

    def __init__(
        self,
        target_sample_rate: int = 16000,
        segment_duration: float = 4.0,
        segment_overlap: float = 0.3,
        silence_threshold: float = 0.01,
        silence_duration: float = 0.5,
        enable_denoise: bool = True,
        high_freq_denoising_cutoff: float = 50.0,
        denoise_order: int = 5,
        max_workers: int | None = None,
        max_seconds_per_file: float | None = 60.0,
        max_segments_per_file: int | None = 500,
        min_segment_seconds: float = 0.5,
        min_chunk_seconds: float = 1.0,
        prefer_backend: str | None = None,
    ) -> None:
        self.target_sample_rate = int(target_sample_rate)
        self.segment_duration = float(segment_duration)
        self.segment_overlap = float(segment_overlap)
        self.silence_threshold = float(silence_threshold)
        self.silence_duration = float(silence_duration)

        self.enable_denoise = bool(enable_denoise)
        self.high_freq_denoising_cutoff = float(high_freq_denoising_cutoff)
        self.denoise_order = int(denoise_order)

        self.max_workers = max_workers
        self.max_seconds_per_file = None if max_seconds_per_file is None else float(max_seconds_per_file)
        self.max_segments_per_file = max_segments_per_file
        self.min_segment_seconds = float(min_segment_seconds)
        self.min_chunk_seconds = float(min_chunk_seconds)
        self.prefer_backend = prefer_backend

        self._resampler_cache: dict[tuple[int, int], torchaudio.transforms.Resample] = {}

    def preprocess(self, data: UnprocessedTrainingData, quiet: bool = False) -> SegmentedAudio:
        audio_files = self._find_audio_files(data.audio_dir_path)
        if not audio_files:
            return SegmentedAudio(segments=[])

        all_segments: list[AudioSegment] = []

        max_workers = self.max_workers or min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_file = {ex.submit(self.preprocess_file, fp): fp for fp in audio_files}

            if quiet:
                for fut in as_completed(future_to_file):
                    self._process_future(fut, future_to_file[fut], all_segments)
            else:
                with tqdm(total=len(audio_files), desc="Preprocess (files)") as pbar:
                    for fut in as_completed(future_to_file):
                        ok = self._process_future(fut, future_to_file[fut], all_segments)
                        if ok:
                            pbar.set_postfix({"segments": len(all_segments)})
                        pbar.update(1)

        return SegmentedAudio(segments=all_segments)

    def _process_future(
        self, future: Future[SegmentedAudio], file_path: str, all_segments: list[AudioSegment]
    ) -> bool:
        try:
            seg = future.result()
            all_segments.extend(seg.segments)
            return True
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False

    def _find_audio_files(self, directory: str) -> list[str]:
        out: list[str] = []
        for root, _, files in os.walk(directory):
            for f in files:
                if Path(f).suffix.lower() in self.AUDIO_EXTENSIONS:
                    out.append(os.path.join(root, f))
        out.sort()
        return out

    def preprocess_file(self, file_path: str) -> SegmentedAudio:
        audio, sr = self._load_audio(file_path)
        if audio.size == 0:
            return SegmentedAudio(segments=[])
        
        if self.max_seconds_per_file is not None:
            max_n = int(self.max_seconds_per_file * sr)
            if audio.size > max_n:
                audio = audio[:max_n]

        audio = self._sanitize(audio)

        if self.enable_denoise:
            audio = self._denoise_audio(audio, sr)

        splits = self._split_on_silence(audio, sr)

        segments: list[AudioSegment] = []
        seg_count = 0

        for split_audio in splits:
            for chunk in self._create_chunks(split_audio, sr):
                chunk = self._sanitize(chunk)
                if chunk.size == 0:
                    continue

                chunk = self._normalize_volume(chunk)
                chunk = self._resample(chunk, sr, self.target_sample_rate)

                segments.append(AudioSegment(audio=chunk, sample_rate=self.target_sample_rate))
                seg_count += 1

                if self.max_segments_per_file is not None and seg_count >= self.max_segments_per_file:
                    return SegmentedAudio(segments=segments)

        return SegmentedAudio(segments=segments)

    def _load_audio(self, file_path: str) -> tuple[np.ndarray, int]:
        backend = self.prefer_backend

        try:
            sig = inspect.signature(torchaudio.load)
            if backend is not None and "backend" in sig.parameters:
                waveform, sr = torchaudio.load(file_path, backend=backend)
            else:
                waveform, sr = torchaudio.load(file_path)
        except TypeError:
            waveform, sr = torchaudio.load(file_path)

        if waveform.dim() != 2:
            raise ValueError("Expected torchaudio.load to return [channels, n_samples].")

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            waveform = waveform[:1]

        audio = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)

        mx = float(np.abs(audio).max()) if audio.size else 0.0
        if mx > 1.0:
            audio = audio / mx

        return audio, int(sr)

    @staticmethod
    def _sanitize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return x

    def _denoise_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        nyq = float(sample_rate) / 2.0
        cutoff = self.high_freq_denoising_cutoff / nyq
        if cutoff <= 0.0 or cutoff >= 1.0:
            return audio

        sos = signal.butter(self.denoise_order, cutoff, btype="high", output="sos")
        return signal.sosfiltfilt(sos, audio).astype(np.float32)

    def _split_on_silence(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        min_silence = int(self.silence_duration * sample_rate)
        if min_silence <= 0:
            return [audio]

        silent = (np.abs(audio) < self.silence_threshold)

        diff = np.diff(silent.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if silent.size and silent[0]:
            starts = np.concatenate([np.array([0], dtype=np.int64), starts])
        if silent.size and silent[-1]:
            ends = np.concatenate([ends, np.array([len(audio)], dtype=np.int64)])

        if starts.size == 0 or ends.size == 0:
            return [audio]

        keep = (ends - starts) >= min_silence
        if not np.any(keep):
            return [audio]

        silences = list(zip(starts[keep].tolist(), ends[keep].tolist()))

        splits: list[np.ndarray] = []
        last = 0
        for s0, s1 in silences:
            if s0 > last:
                splits.append(audio[last:s0])
            last = s1
        if last < audio.size:
            splits.append(audio[last:])

        min_seg = int(self.min_segment_seconds * sample_rate)
        splits = [s for s in splits if s.size >= min_seg]
        return splits or [audio]

    def _create_chunks(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        seg_n = int(self.segment_duration * sample_rate)
        ov_n = int(self.segment_overlap * sample_rate)
        hop = max(1, seg_n - ov_n)

        if seg_n <= 0:
            return [audio]

        min_n = int(self.min_chunk_seconds * sample_rate)

        out: list[np.ndarray] = []
        start = 0
        n = int(audio.size)

        while start < n:
            end = min(start + seg_n, n)
            chunk = audio[start:end]
            if chunk.size < min_n:
                break
            out.append(chunk)
            start += hop

        return out

    @staticmethod
    def _normalize_volume(audio: np.ndarray) -> np.ndarray:
        mx = float(np.abs(audio).max()) if audio.size else 0.0
        return (audio / mx).astype(np.float32) if mx > 0 else audio.astype(np.float32)

    def _get_resampler(self, orig: int, new: int) -> torchaudio.transforms.Resample:
        key = (int(orig), int(new))
        r = self._resampler_cache.get(key)
        if r is None:
            r = torchaudio.transforms.Resample(orig_freq=orig, new_freq=new)
            self._resampler_cache[key] = r
        return r

    def _resample(self, audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        if int(original_sr) == int(target_sr):
            return audio.astype(np.float32)

        x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        r = self._get_resampler(original_sr, target_sr)
        y = r(x).squeeze(0).detach().cpu().numpy()
        return y.astype(np.float32)
