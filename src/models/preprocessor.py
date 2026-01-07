import os
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

import numpy as np
import torchaudio
import torch
from scipy import signal
from tqdm import tqdm

from src.data_models.data_models import UnprocessedTrainingData, AudioSegment, SegmentedAudio


class Preprocessor:
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}

    def __init__(
        self,
        target_sample_rate: int = 16000,
        segment_duration: float = 4.0,
        segment_overlap: float = 0.3,
        silence_threshold: float = 0.01,
        silence_duration: float = 0.5,
        high_freq_denoising_cutoff: float = 50.0,
        denoise_order: int = 5,
        max_workers: int | None = None,
    ):
        self.target_sample_rate = target_sample_rate
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.high_freq_denoising_cutoff = high_freq_denoising_cutoff
        self.denoise_order = denoise_order
        self.max_workers = max_workers

    def preprocess(self, data: UnprocessedTrainingData, quiet: bool = False) -> SegmentedAudio:
        audio_files = self._find_audio_files(data.audio_dir_path)
        if not audio_files:
            return SegmentedAudio(segments=[])

        all_segments: list[AudioSegment] = []

        # ThreadPool avoids torch/torchaudio + multiprocessing hangs and pickling overhead.
        max_workers = self.max_workers or min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.preprocess_file, file_path): file_path
                for file_path in audio_files
            }

            if quiet:
                for future in as_completed(future_to_file):
                    self._process_future(future, future_to_file[future], all_segments)
            else:
                with tqdm(total=len(audio_files), desc="Processing audio files") as pbar:
                    for future in as_completed(future_to_file):
                        ok = self._process_future(future, future_to_file[future], all_segments)
                        if ok:
                            pbar.set_postfix({"segments": len(all_segments)})
                        pbar.update(1)

        return SegmentedAudio(segments=all_segments)

    def _process_future(
        self, future: Future[SegmentedAudio], file_path: str, all_segments: list[AudioSegment]
    ) -> bool:
        try:
            segmented_audio = future.result()
            all_segments.extend(segmented_audio.segments)
            return True
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False

    def _find_audio_files(self, directory: str) -> list[str]:
        audio_files: list[str] = []
        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in self.AUDIO_EXTENSIONS:
                    audio_files.append(os.path.join(root, file))
        return audio_files

    def preprocess_file(self, file_path: str) -> SegmentedAudio:
        audio, original_sample_rate = self._load_audio(file_path)

        audio = self._denoise_audio(audio, original_sample_rate)
        splits = self._split_on_silence(audio, original_sample_rate)

        segments: list[AudioSegment] = []
        for split_audio in splits:
            for chunk in self._create_chunks(split_audio, original_sample_rate):
                chunk = self._normalize_volume(chunk)
                chunk = self._resample(chunk, original_sample_rate, self.target_sample_rate)
                segments.append(AudioSegment(audio=chunk, sample_rate=self.target_sample_rate))

        return SegmentedAudio(segments=segments)

    def _load_audio(self, file_path: str) -> tuple[np.ndarray, int]:
        waveform, sample_rate = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        audio = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)

        max_val = float(np.abs(audio).max())
        if max_val > 1.0:
            audio = audio / max_val

        return audio, sample_rate

    def _denoise_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        nyquist = sample_rate / 2.0
        cutoff = self.high_freq_denoising_cutoff / nyquist
        if cutoff >= 1.0 or cutoff <= 0.0:
            return audio

        b, a = signal.butter(self.denoise_order, cutoff, btype="high", analog=False)
        denoised = signal.filtfilt(b, a, audio).astype(np.float32)
        return denoised

    def _split_on_silence(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        min_silence_samples = int(self.silence_duration * sample_rate)
        if min_silence_samples <= 0:
            return [audio]

        silent = (np.abs(audio) < self.silence_threshold)

        # Vectorized run detection: find contiguous True regions.
        diff = np.diff(silent.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if silent[0]:
            starts = np.concatenate([np.array([0], dtype=np.int64), starts])
        if silent[-1]:
            ends = np.concatenate([ends, np.array([len(audio)], dtype=np.int64)])

        if starts.size == 0 or ends.size == 0:
            return [audio]

        lengths = ends - starts
        keep = lengths >= min_silence_samples
        if not np.any(keep):
            return [audio]

        silence_regions = list(zip(starts[keep].tolist(), ends[keep].tolist()))

        splits: list[np.ndarray] = []
        last_end = 0
        for s0, s1 in silence_regions:
            if s0 > last_end:
                splits.append(audio[last_end:s0])
            last_end = s1

        if last_end < len(audio):
            splits.append(audio[last_end:])

        # Drop ultra-short remnants
        min_segment_samples = int(0.5 * sample_rate)
        splits = [s for s in splits if s.size >= min_segment_samples]
        return splits or [audio]

    def _create_chunks(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        segment_samples = int(self.segment_duration * sample_rate)
        overlap_samples = int(self.segment_overlap * sample_rate)
        hop_samples = max(1, segment_samples - overlap_samples)

        if segment_samples <= 0:
            return [audio]

        chunks: list[np.ndarray] = []
        start = 0
        while start < len(audio):
            end = min(start + segment_samples, len(audio))
            chunk = audio[start:end]

            # Skip final tiny tail
            if chunk.size < sample_rate:
                break

            chunks.append(chunk)
            start += hop_samples

        return chunks

    def _normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        max_val = float(np.abs(audio).max())
        return (audio / max_val).astype(np.float32) if max_val > 0 else audio.astype(np.float32)

    def prepare_inference_audio(self, audio: np.ndarray, sample_rate: int) -> AudioSegment:
        audio = self._normalize_volume(audio)
        audio = self._denoise_audio(audio, sample_rate)
        if sample_rate != self.target_sample_rate:
            audio = self._resample(audio, sample_rate, self.target_sample_rate)
            sample_rate = self.target_sample_rate
        return AudioSegment(audio=audio, sample_rate=sample_rate)

    def _resample(self, audio: np.ndarray, original_sample_rate: int, target_sample_rate: int) -> np.ndarray:
        if original_sample_rate == target_sample_rate:
            return audio.astype(np.float32)

        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate, new_freq=target_sample_rate
        )
        resampled = resampler(audio_tensor).squeeze(0).detach().cpu().numpy()
        return resampled.astype(np.float32)
