# %%
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from scipy.io.wavfile import write
from tqdm import tqdm

from Evaluate.Normalizer import filterAndNormalize

# %%
model_variant = "atco2-asr-atcosim"
dts = f"jlvdoorn/{model_variant}"
spl = "train+validation"
wsp = "whisper-large-v3"

# %% [markdown]
# # Load Timestamped Prediction Dataset

# %%
data_folder = Path("./data")
df_prediction = pd.read_pickle(
    data_folder / f"{model_variant}-train+validation-{wsp}-{model_variant}.pickle"
)
for col in ["ref", "hyp-clean", "hyp-prmpt"]:
    df_prediction[f"{col}-norm"] = df_prediction.apply(
        lambda x: filterAndNormalize(x[f"{col}"]), axis=1
    )

# %% [markdown]
# # Save Data


# %%
def seconds_to_timecode(seconds):
    return "{0:02d}:{1:02d}:{2:02d},{3:03d}".format(
        int(seconds // 3600),
        int(seconds // 60 % 60),
        int(seconds % 60),
        int((100.0 * seconds) % 100),
    )


# %%
audio_folder = Path("./audio")
audio_folder.mkdir(exist_ok=True)

for s_idx, s in tqdm(enumerate(spl.split("+"))):
    ds_audio = load_dataset(dts)[s]
    ds_prediction = df_prediction.query("split == @s")

    for i in tqdm(range(len(ds_prediction))):
        file_name = f"audio_{model_variant}_{s}_{i:04d}"
        # save audio data
        sample = ds_audio[i]
        audio = sample["audio"]
        write(
            audio_folder / f"{file_name}.wav",
            audio["sampling_rate"],
            audio["array"],
        )

        # save subtitle data
        sample = ds_prediction.iloc[i]
        line_counter = 0
        subtitle_file = audio_folder / f"{file_name}.srt"
        label_file = audio_folder / f"{file_name}.txt"
        subtitle_file.unlink(missing_ok=True)
        label_file.unlink(missing_ok=True)
        for seq in sample["words-clean"]:
            for word in seq["words"]:
                start = word["start"]
                end = word["end"]
                subtext = word["word"]
                with open(subtitle_file, "at") as file:
                    file.write(f"{(line_counter := line_counter + 1)}\n")
                    file.write(
                        f"{seconds_to_timecode(start)} --> {seconds_to_timecode(end)}\n"
                    )
                    file.write(subtext.strip())
                    file.write("\n\n")

                with open(label_file, "at") as file:
                    file.write(f"{start}\t{end}\t{subtext}\n")
