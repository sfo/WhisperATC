from pathlib import Path

import scipy.io.wavfile as wavfile
import torch
from datasets import load_dataset


class AudioDataset:
    def __init__(self, dts: str, spl: str, device: str | torch.device):
        super().__init__()
        self._ds_audio = load_dataset(dts)[spl]  # type: ignore
        self._device = device

    def get_audio_sample(
        self, sample_index: int, waveform2d: bool
    ) -> tuple[str, torch.Tensor, int, Path, bytes]:
        sample = self._ds_audio[sample_index]
        waveform = torch.tensor(
            sample["audio"]["array"], device=self._device, dtype=torch.float32
        )
        if waveform2d:
            waveform = torch.atleast_2d(waveform)
        sampling_rate = sample["audio"]["sampling_rate"]
        TRANSCRIPT = sample["text"].strip()

        audio_file_path = Path(sample["audio"]["path"])

        audio_array = sample["audio"]["array"]

        return TRANSCRIPT, waveform, sampling_rate, audio_file_path, audio_array

    def export_audio(self, sample_idx, output_path: str | Path = ".") -> Path:
        _, _, sampling_rate, file_name, audio_array = self.get_audio_sample(
            sample_idx, False
        )
        audio_file_path = Path(output_path) / file_name
        wavfile.write(audio_file_path, sampling_rate, audio_array)
        return audio_file_path
