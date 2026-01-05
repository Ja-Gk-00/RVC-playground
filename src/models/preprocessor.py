import os
from pathlib import Path
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

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
        silence_duration: float = 5.0,
        high_freq_denoising_cutoff: float = 50.0,
        denoise_order: int = 5,
        max_workers: int | None = None,
    ):
        """
        Initialize the preprocessor.

        Args:
            target_sample_rate: Target sample rate in Hz (16000 for ContentVec)
            segment_duration: Duration of each segment in seconds
            segment_overlap: Overlap between consecutive segments in seconds
            silence_threshold: Amplitude threshold for silence detection (0-1)
            silence_duration: Minimum silence duration to split at, in seconds
            high_freq_denoising_cutoff: Cutoff frequency for high-pass denoising filter in Hz
            denoise_order: Order of the Butterworth filter for denoising
            max_workers: Maximum number of parallel workers (None = CPU count)
        """
        self.target_sample_rate = target_sample_rate
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.high_freq_denoising_cutoff = high_freq_denoising_cutoff
        self.denoise_order = denoise_order
        self.max_workers = max_workers

    def preprocess(self, data: UnprocessedTrainingData, quiet: bool = False) -> SegmentedAudio:
        """
        Preprocess all audio files in a directory (including subdirectories).

        Args:
            data: Training data specification with audio directory path

        Returns:
            SegmentedAudio containing all processed segments from all files
        """
        audio_files = self._find_audio_files(data.audio_dir_path)

        if not audio_files:
            return SegmentedAudio(segments=[])

        all_segments = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
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
                        if self._process_future(future, future_to_file[future], all_segments):
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
        """
        Find all audio files in directory and subdirectories.

        Args:
            directory: Root directory to search

        Returns:
            List of paths to audio files
        """
        audio_files = []

        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in self.AUDIO_EXTENSIONS:
                    audio_files.append(os.path.join(root, file))

        return audio_files

    def preprocess_file(self, file_path: str) -> SegmentedAudio:
        """
        Preprocess a single audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            SegmentedAudio containing processed audio segments
        """
        audio, original_sample_rate = self._load_audio(file_path)
        audio = self._denoise_audio(audio, original_sample_rate)
        silence_splits = self._split_on_silence(audio, original_sample_rate)

        all_chunks = []
        for split_audio in silence_splits:
            chunks = self._create_chunks(split_audio, original_sample_rate)
            all_chunks.extend(chunks)

        segments = []
        for chunk in all_chunks:
            normalized_chunk = self._normalize_volume(chunk)
            resampled_chunk = self._resample(
                normalized_chunk, original_sample_rate, self.target_sample_rate
            )

            segments.append(
                AudioSegment(
                    audio=resampled_chunk,
                    sample_rate=self.target_sample_rate,
                )
            )

        return SegmentedAudio(segments)

    def _load_audio(self, file_path: str) -> tuple[np.ndarray, int]: # ([n_samples], sample_rate)
        """
        Load audio file and normalize to float32 in range [-1, 1].

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (normalized_audio [n_samples], sample_rate)
        """
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        audio = waveform.squeeze().numpy().astype(np.float32)

        # Normalize to [-1, 1] if not already
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        return audio, sample_rate

    def _denoise_audio(
        self,
        audio: np.ndarray,  # [n_samples]
        sample_rate: int,
    ) -> np.ndarray:  # [n_samples]
        """
        Apply high-pass Butterworth filter to remove low-frequency noise.

        Args:
            audio: Input audio signal [n_samples]
            sample_rate: Sample rate of the audio in Hz

        Returns:
            Denoised audio signal [n_samples]
        """
        nyquist = sample_rate / 2
        normalized_cutoff = self.high_freq_denoising_cutoff / nyquist

        # Skip filtering if cutoff is invalid
        if normalized_cutoff >= 1.0:
            return audio

        b, a = signal.butter(self.denoise_order, normalized_cutoff, btype="high", analog=False)

        denoised = signal.filtfilt(b, a, audio)
        return denoised.astype(np.float32)

    def _split_on_silence(
        self,
        audio: np.ndarray,  # [n_samples]
        sample_rate: int,
    ) -> list[np.ndarray]:  # list of [n_segment_samples_i]
        """
        Split audio at silence gaps longer than silence_duration.

        Args:
            audio: Input audio signal [n_samples]
            sample_rate: Sample rate of the audio in Hz

        Returns:
            List of audio segments [n_segments] each with shape [n_segment_samples]
        """
        min_silence_samples = int(self.silence_duration * sample_rate)
        is_silent = np.abs(audio) < self.silence_threshold

        silence_regions = []
        start = None

        for i, silent in enumerate(is_silent):
            if silent and start is None:
                start = i
            elif not silent and start is not None:
                if i - start >= min_silence_samples:
                    silence_regions.append((start, i))
                start = None

        # Handle case where audio ends in silence - append the remaining audio
        if start is not None and len(audio) - start >= min_silence_samples:
            silence_regions.append((start, len(audio)))

        if not silence_regions:
            return [audio]

        splits = []
        last_end = 0

        for silence_start, silence_end in silence_regions:
            if silence_start > last_end:
                splits.append(audio[last_end:silence_start])
            last_end = silence_end

        if last_end < len(audio):
            splits.append(audio[last_end:])

        # Filter out very short segments (< 0.5 seconds)
        min_segment_samples = int(0.5 * sample_rate)
        splits = [s for s in splits if len(s) >= min_segment_samples]

        return splits

    def _create_chunks(
        self,
        audio: np.ndarray,  # [n_samples]
        sample_rate: int,
    ) -> list[np.ndarray]:  # list of [n_chunk_samples_i]
        """
        Create fixed-duration chunks with overlap from audio segment.

        Args:
            audio: Input audio signal [n_samples]
            sample_rate: Sample rate of the audio in Hz

        Returns:
            List of audio chunks [n_chunks] each with shape [n_chunk_samples]
        """
        segment_samples = int(self.segment_duration * sample_rate)
        overlap_samples = int(self.segment_overlap * sample_rate)
        hop_samples = segment_samples - overlap_samples

        chunks = []
        start = 0

        while start < len(audio):
            end = min(start + segment_samples, len(audio))
            chunk = audio[start:end]

            # Skip if chunk is too short (< 1 second)
            if len(chunk) < sample_rate:
                break

            chunks.append(chunk)
            start += hop_samples

        return chunks

    def _normalize_volume(
        self,
        audio: np.ndarray,  # [n_samples]
    ) -> np.ndarray:  # [n_samples]
        """
        Normalize volume of audio segment to [-1, 1] range.

        Args:
            audio: Input audio signal [n_samples]

        Returns:
            Volume-normalized audio [n_samples]
        """
        max_val = np.abs(audio).max()
        return audio / max_val if max_val > 0 else audio

    def prepare_inference_audio(
        self,
        audio: np.ndarray,  # [n_samples]
        sample_rate: int,
    ) -> AudioSegment:
        """
        Prepare audio for inference by applying normalization, denoising, and resampling.

        Args:
            audio: Input audio signal [n_samples]
            sample_rate: Sample rate of the audio in Hz

        Returns:
            AudioSegment ready for feature extraction
        """
        audio = self._normalize_volume(audio)
        audio = self._denoise_audio(audio, sample_rate)
        
        if sample_rate != self.target_sample_rate:
            audio = self._resample(audio, sample_rate, self.target_sample_rate)
            sample_rate = self.target_sample_rate
        
        return AudioSegment(audio=audio, sample_rate=sample_rate)

    def _resample(
        self,
        audio: np.ndarray,  # [n_samples]
        original_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:  # [n_resampled_samples]
        """
        Resample audio to target sample rate.

        Args:
            audio: Input audio signal [n_samples]
            original_sample_rate: Original sample rate in Hz
            target_sample_rate: Target sample rate in Hz

        Returns:
            Resampled audio [n_resampled_samples]
        """
        if original_sample_rate == target_sample_rate:
            return audio

        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate, new_freq=target_sample_rate
        )
        resampled = resampler(audio_tensor).squeeze().numpy()

        return resampled.astype(np.float32)
