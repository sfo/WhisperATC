# %%
from pathlib import Path

import torch
from datasets import load_dataset
from scipy.io.wavfile import write

from alignment_utils import word_to_audacity_label

# %%
torch.set_num_threads(1)

# %%
model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
(get_speech_timestamps, _, read_audio, _, _) = utils

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_variant = "atcosim"
dts = f"jlvdoorn/{model_variant}"
spl = "train"
ds_audio = load_dataset(dts)[spl]
sample_index = 42

sample = ds_audio[sample_index]
waveform = torch.atleast_2d(
    torch.tensor(sample["audio"]["array"], device=device, dtype=torch.float32)
)
sampling_rate = sample["audio"]["sampling_rate"]

# %%
speech_timestamps = get_speech_timestamps(waveform, model, sampling_rate=sampling_rate)

# %%
audio_file_path = Path(sample["audio"]["path"])
with open(audio_file_path.with_suffix(".vad.txt"), "tw") as file:
    file.write(
        "\n".join(
            [
                word_to_audacity_label(
                    waveform,
                    [{k: v} for k, v in ts.items()],
                    waveform.size(1),
                    "activity",
                    sampling_rate,
                )
                for ts in speech_timestamps
            ]
        )
    )

write(
    audio_file_path,
    sampling_rate,
    sample["audio"]["array"],
)

# %%
