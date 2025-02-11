# %%
import os
from pathlib import Path

import dotenv
import torch
from audio_utils import AudioDataset
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

# %%
if (secrets_file := dotenv.find_dotenv("secrets.env", usecwd=True)) == "":
    raise FileNotFoundError("Could not find environment file holding the API key!")
else:
    dotenv.load_dotenv(secrets_file)

DG_KEY = os.environ["DEEPGRAM_API_KEY"]

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

# %% MODEL SPECIFIC STUFF
#########################

# %% PREPARE TRANSCRIPT
# TODO - for testing, use a single sample. Rework into a loop.
for sample_index in (1, 2, 3, 26, 42, 100):
    audio_file_path = audio_dataset.export_audio(sample_index, audio_path)
    print("Saved audio file to", audio_file_path)

    with open(audio_file_path, "rb") as file:
        source: FileSource = {"buffer": file.read()}

    try:
        deepgram = DeepgramClient(DG_KEY)
        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
        )
        response = deepgram.listen.prerecorded.v("1").transcribe_file(source, options)
        with open(audio_file_path.with_suffix(".dg.json"), "w") as transcript_file:
            transcript_file.write(response.to_json(indent=4))
        print("Done.")
    except Exception as e:
        print(f"Exception; {e}")
