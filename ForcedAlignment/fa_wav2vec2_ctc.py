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
from pathlib import Path

import torch
import torchaudio
from alignment_utils import (
    aggregate_words,
    align_tokens,
    get_emission,
    whisper_transcript_to_audacity_label,
)
from audio_utils import AudioDataset

# %%
dts = "jlvdoorn/atcosim"
# spl = "train+validation"
spl = "train"
wsp = "large-v3"
mdl = f"jlvdoorn/whisper-{wsp}-atcosim"

print("Dataset: ", dts)
print("Model  : ", mdl)
print("Split  : ", spl)
print("Whisper: ", wsp)

# %%
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device: ", device)

# %%
audio_dataset = AudioDataset(dts, "train", device)
audio_path = Path("./audio")
audio_path.mkdir(exist_ok=True)

# %% MODEL SPECIFIC STUFF
#########################

# %% SETUP MODEL
bundle = torchaudio.pipelines.MMS_FA
LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)

# %% PREPARE TRANSCRIPT
# TODO - for testing, use a single sample. Rework into a loop.
for sample_index in (1, 2, 3, 26, 42, 100):
    TRANSCRIPT, waveform, sampling_rate, audio_file_name, _ = (
        audio_dataset.get_audio_sample(sample_index, waveform2d=True)
    )
    TRANSCRIPT = TRANSCRIPT.split()
    tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word]

    ## %% PERFORM ALIGNMENT
    # TODO - check if emission reflect actual transcript and base statistics only on accurate predictions
    # TODO - apply WhisperATC's fine tuning to the wav2vec2 model
    emission = get_emission(waveform, bundle, device)
    token_spans = align_tokens(emission, tokenized_transcript, device)
    word_spans = aggregate_words(token_spans, TRANSCRIPT)

    ## %% SAVE RESULTS
    audio_dataset.export_audio(sample_index, audio_path)

    num_frames = emission.size(1)
    with open(audio_path / audio_file_name.with_suffix(".w2v2.txt"), "tw") as file:
        whisper_transcript_to_audacity_label(
            waveform, word_spans, num_frames, TRANSCRIPT, sampling_rate, file
        )

# %%
