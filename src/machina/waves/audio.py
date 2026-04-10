from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Optional

import numpy as np
from scipy.io import wavfile


class Audio:
    """
    Represents a digital audio signal.

    Args:
        samples (np.ndarray): A NumPy array of audio samples.
            For mono, shape is (n_samples,).
            For stereo, shape is (n_samples, 2).
        sample_rate (int): The number of samples per second.

    Attributes:
        samples (np.ndarray): The raw audio samples.
        sample_rate (int): The sampling rate of the audio.
        duration (float): The duration of the audio in seconds.
        channels (int): The number of audio channels (1 for mono, 2 for stereo).
    """

    def __init__(self, samples: np.ndarray, sample_rate: int):
        if not isinstance(samples, np.ndarray):
            raise TypeError("Samples must be a NumPy array.")
        if samples.ndim > 2:
            raise ValueError("Samples must be a 1D (mono) or 2D (stereo) array.")

        self._samples = samples
        self._sample_rate = sample_rate

    def __repr__(self) -> str:
        return f"Audio(duration={self.duration:.2f}s, sample_rate={self.sample_rate}, channels={self.channels})"

    @property
    def data(self) -> dict:
        """Returns a dictionary representation of the audio data."""
        return {
            "type": "audio",
            "samples": self._samples.tolist(),
            "sample_rate": self._sample_rate,
        }

    @property
    def samples(self) -> np.ndarray:
        """The raw audio samples."""
        return self._samples

    @property
    def sample_rate(self) -> int:
        """The sampling rate of the audio."""
        return self._sample_rate

    @property
    def duration(self) -> float:
        """The duration of the audio in seconds."""
        return len(self._samples) / self._sample_rate

    @property
    def channels(self) -> int:
        """The number of audio channels (1 for mono, 2 for stereo)."""
        return self._samples.ndim

    @classmethod
    def from_data(cls, data: dict) -> Audio:
        """Creates an Audio instance from a dictionary."""
        if data.get("type") != "audio":
            raise ValueError("Data type must be 'audio'")
        samples = np.array(data["samples"])
        return cls(samples, data["sample_rate"])

    @classmethod
    def from_file(cls, file_path: str) -> Audio:
        """
        Creates an Audio instance by loading a .wav file.

        Args:
            file_path: The path to the .wav file.

        Returns:
            An Audio instance with the loaded data.
        """
        sample_rate, samples = wavfile.read(file_path)
        return cls(samples, sample_rate)

    @classmethod
    def from_silence(
        cls, duration: float, sample_rate: int = 44100, channels: int = 1
    ) -> Audio:
        """
        Creates a silent audio clip.

        Args:
            duration: The duration of the silence in seconds.
            sample_rate: The desired sample rate.
            channels: The number of channels (1 for mono, 2 for stereo).

        Returns:
            An Audio instance representing silence.
        """
        num_samples = int(duration * sample_rate)
        shape = (num_samples, channels) if channels > 1 else (num_samples,)
        # Use int16 for standard WAV file compatibility
        silent_samples = np.zeros(shape, dtype=np.int16)
        return cls(silent_samples, sample_rate)

    def copy(self) -> Audio:
        """Creates a deep copy of the current Audio instance."""
        return Audio(self._samples.copy(), self._sample_rate)

    def save(self, file_path: str) -> None:
        """Saves the current audio to a .wav file."""
        wavfile.write(file_path, self._sample_rate, self._samples)

    def play(self):
        """Plays the current audio using the default system audio player."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        self.save(temp_path)

        try:
            if sys.platform.startswith("darwin"):  # macOS
                subprocess.call(("open", temp_path))
            elif os.name == "nt":  # Windows
                os.startfile(temp_path)
            elif os.name == "posix":  # Linux/Unix
                subprocess.call(("xdg-open", temp_path))
        except Exception as e:
            print(f"Could not play audio: {e}")

    def crop(self, start_time: float, end_time: float) -> Audio:
        """
        Crops the audio to the specified time interval in place.

        Args:
            start_time: The start time in seconds.
            end_time: The end time in seconds.
        """
        start_sample = int(start_time * self._sample_rate)
        end_sample = int(end_time * self._sample_rate)
        self._samples = self._samples[start_sample:end_sample]
        return self

    def reverse(self) -> Audio:
        """Reverses the audio in place."""
        self._samples = np.flip(self._samples, axis=0)
        return self

    def change_speed(self, factor: float) -> Audio:
        """
        Changes the speed of the audio without changing the pitch.
        Note: This is a simple implementation and may affect audio quality.

        Args:
            factor: The speed change factor. > 1.0 for faster, < 1.0 for slower.
        """
        indices = np.round(np.arange(0, len(self._samples), factor)).astype(int)
        indices = indices[indices < len(self._samples)]
        self._samples = self._samples[indices]
        return self

    def overlay(self, other: Audio, start_time: float = 0) -> Audio:
        """
        Overlays another audio clip onto the current one.

        Args:
            other: The audio clip to overlay.
            start_time: The time in seconds to start the overlay.

        Returns:
            The current Audio instance with the other audio overlaid.
        """
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rates of both audio clips must match.")

        start_sample = int(start_time * self.sample_rate)
        end_sample = start_sample + len(other.samples)

        # Ensure self._samples is long enough
        if end_sample > len(self._samples):
            padding = np.zeros(
                (end_sample - len(self._samples),)
                if self.channels == 1
                else (end_sample - len(self._samples), self.channels),
                dtype=self._samples.dtype,
            )
            self._samples = np.concatenate((self._samples, padding))

        # Mix audio (simple addition, could lead to clipping)
        self._samples[start_sample:end_sample] += other.samples
        return self
