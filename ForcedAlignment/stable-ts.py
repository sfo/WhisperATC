# %%
%load_ext autoreload
%autoreload 2

# %%
import stable_whisper
import torch
from audio_utils import AudioDataset
from model_loader import ModelLoader
from scipy.io import wavfile
import scipy.signal as sps


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

model_loader = ModelLoader(base_model_loader=stable_whisper.load_model)
model = model_loader.load_model(wsp, mdl)
# model = stable_whisper.load_model(wsp)

# %%
audio_dataset = AudioDataset(dts, "train", device)

# %%
sample_index = 69
TRANSCRIPT, waveform, sampling_rate, audio_file_name, audio_array = (
    audio_dataset.get_audio_sample(sample_index, waveform2d=False)
)

new_rate = 16_000
number_of_samples = round(len(waveform) * float(new_rate) / sampling_rate)
resampled = sps.resample(waveform, number_of_samples)
waveform = resampled

# %%
initial_alignment = model.align(waveform, TRANSCRIPT, language="en")
aligned_alignment = model.align(waveform, initial_alignment, language="en")
refined_alignment = model.refine(waveform, aligned_alignment)

# %%
audio_dataset.export_audio(sample_index, ".")


def save_alignment(label, alignment):
    alignment.to_srt_vtt(audio_file_name.with_suffix(f".{label}.srt").as_posix())
    with open(audio_file_name.with_suffix(f".{label}.txt"), "wt") as file:
        for segment in alignment.to_dict()['segments']:
            for word in segment['words']:
                file.write(f"{word['start']}\t{word['end']}\t{word['word'].strip()}\n")


save_alignment("initial", initial_alignment)
save_alignment("aligned", aligned_alignment)
save_alignment("refined", refined_alignment)

initial_alignment.adjust_by_result(aligned_alignment)
save_alignment("adjusted", initial_alignment)

# %%