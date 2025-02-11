# %%
from pathlib import Path

import scipy.signal as sps
import stable_whisper
import torch
from alignment_utils import stable_ts_alignment_to_audacity_label
from audio_utils import AudioDataset
from model_loader import ModelLoader

# %%
dts = "jlvdoorn/atcosim"
spl = "train+validation"
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


def save_alignment(audio_file_name, label, alignment):
    alignment.to_srt_vtt(
        (audio_path / audio_file_name.with_suffix(f".{label}.srt")).as_posix()
    )
    with open(audio_path / audio_file_name.with_suffix(f".{label}.txt"), "wt") as file:
        stable_ts_alignment_to_audacity_label(alignment, file)


# %% MODEL SPECIFIC STUFF
#########################

# %% SETUP MODEL
model_loader = ModelLoader(base_model_loader=stable_whisper.load_model)
model = model_loader.load_model(wsp, mdl)

# %% PREPARE TRANSCRIPT
# TODO - for testing, use a single sample. Rework into a loop.
for sample_index in (1, 2, 3, 26, 42, 100):
    TRANSCRIPT, waveform, sampling_rate, audio_file_name, _ = (
        audio_dataset.get_audio_sample(sample_index, waveform2d=False)
    )

    ## %% PREPARE AUDIO
    new_rate = 16_000
    number_of_samples = round(len(waveform) * float(new_rate) / sampling_rate)
    resampled = sps.resample(waveform, number_of_samples)
    waveform = resampled

    ## %% PERFORM ALIGNMENT
    initial_alignment = model.align(waveform, TRANSCRIPT, language="en")
    aligned_alignment = model.align(waveform, initial_alignment, language="en")
    refined_alignment = model.refine(waveform, aligned_alignment)

    ## %% SAFE RESULTS
    audio_dataset.export_audio(sample_index, audio_path)

    save_alignment(audio_file_name, "initial", initial_alignment)
    save_alignment(audio_file_name, "aligned", aligned_alignment)
    save_alignment(audio_file_name, "refined", refined_alignment)

    initial_alignment.adjust_by_result(aligned_alignment)
    save_alignment(audio_file_name, "adjusted", initial_alignment)

    # %%
