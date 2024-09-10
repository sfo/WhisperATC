# %% [markdown]
#
# # CTC Forced Alignment using wav2vec2
#
# [Source](https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html)
#
# This script uses a wav2vec2 model to force alignment of existing transcripts to audio.
# It includes a voice recognition step for generating tokens.
#
# Results are OK but not yet usable for running statistical analyses.
#
# %%
import torch
import torchaudio
from alignment_utils import (
    aggregate_words,
    align_tokens,
    get_emission,
    plot_alignments,
    plot_emission,
    word_to_audacity_label,
)
from audio_utils import AudioDataset
from scipy.io.wavfile import write

print(torch.__version__)
print(torchaudio.__version__)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
model_variant = "atcosim"
dts = f"jlvdoorn/{model_variant}"
spl = "train"
ds_audio = AudioDataset(dts, spl, device)
sample_index = 42

TRANSCRIPT, waveform, sampling_rate, audio_file_path, audio_array = (
    ds_audio.get_audio_sample(sample_index, waveform2d=True)
)
TRANSCRIPT = TRANSCRIPT.split()

# %%
bundle = torchaudio.pipelines.MMS_FA
LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)
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

with open(audio_file_path.with_suffix(".txt"), "tw") as file:
    file.write(
        "\n".join(
            [
                word_to_audacity_label(
                    waveform,
                    [{"start": ts.start, "end": ts.end} for ts in word_spans[i]],
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
    audio_array,
)


# %%
plot_alignments(waveform, word_spans, emission, TRANSCRIPT, sampling_rate)

# %%
