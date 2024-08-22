# %%
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset
from scipy.io.wavfile import write

from alignment_utils import (
    aggregate_words,
    align_tokens,
    get_emission,
    plot_alignments,
    plot_emission,
    word_to_audacity_label,
)

print(torch.__version__)
print(torchaudio.__version__)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
model_variant = "atcosim"
dts = f"jlvdoorn/{model_variant}"
spl = "train"
ds_audio = load_dataset(dts)[spl]
sample_index = 2

# %%
bundle = torchaudio.pipelines.MMS_FA
LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)


sample = ds_audio[sample_index]
waveform = torch.atleast_2d(
    torch.tensor(sample["audio"]["array"], device=device, dtype=torch.float32)
)
sampling_rate = sample["audio"]["sampling_rate"]
TRANSCRIPT = sample["text"].strip().split()
tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word]

# %%
# TODO - check if emission reflect actual transcript and base statistics only on accurate predictions
# TODO - apply WhisperATC's fine tuning to the wav2vec2 model
emission = get_emission(waveform, bundle, device)
plot_emission(emission)

token_spans = align_tokens(emission, tokenized_transcript, device)
word_spans = aggregate_words(token_spans, TRANSCRIPT)


# %%
num_frames = emission.size(1)

audio_file_path = Path(sample["audio"]["path"])
with open(audio_file_path.with_suffix(".txt"), "tw") as file:
    file.write(
        "\n".join(
            [
                word_to_audacity_label(
                    waveform,
                    [ {'start': ts.start, 'end': ts.end } for ts in word_spans[i] ],
                    num_frames,
                    word,
                    sampling_rate,
                )
                for i, word in enumerate(TRANSCRIPT)
            ]
        )
    )

write(
    audio_file_path,
    sampling_rate,
    sample["audio"]["array"],
)


# %%
plot_alignments(waveform, word_spans, emission, TRANSCRIPT, sampling_rate)

# %%
