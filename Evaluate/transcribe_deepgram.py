# %%
import io
import os
from pathlib import Path

import dotenv
import pandas as pd
import scipy.io.wavfile as wavfile
from datasets import Audio, load_dataset
from deepgram import DeepgramClient, PrerecordedOptions
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm


def audio_array_to_bytes(sample):
    audio = sample["array"]
    sampling_rate = sample["sampling_rate"]
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    wavfile.write(byte_io, sampling_rate, audio)
    return byte_io.read()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def request_transcript(sample, model):
    return deepgram.listen.rest.v("1").transcribe_file(
        source={
            "buffer": audio_array_to_bytes(sample),
        },
        options=PrerecordedOptions(
            model=model,
            smart_format=True,
        ),
    )


# %%
dts = "jlvdoorn/atcosim"
spl = "validation"
mdl = "nova-3"

print("Dataset: ", dts)
print("Split  : ", spl)
print("Model  : ", mdl)

# %%
if (secrets_file := dotenv.find_dotenv("secrets.env", usecwd=True)) == "":
    raise FileNotFoundError("Could not find environment file holding the API key!")
else:
    dotenv.load_dotenv(secrets_file)

DG_KEY = os.environ["DEEPGRAM_API_KEY"]
deepgram = DeepgramClient(DG_KEY)

# %%
dataset = load_dataset(dts)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# fmt: off
output_file = (dts.split("/")[-1]
    + "-"
    + spl
    + "-"
    + "deepgram"
    + "-"
    + mdl
    + ".pickle"
)
# fmt: on

if Path(output_file).exists():
    df = pd.read_pickle(output_file)
else:
    df = pd.DataFrame()
offset = len(df)


# %%
for s in tqdm(spl.split("+")):
    for i in tqdm(range(offset, len(dataset[s]))):
        sample = dataset[s][i]["audio"]
        response = request_transcript(sample, mdl)
        series = pd.Series(
            {
                "split": s,
                "ref": dataset[s][i]["text"],
                "results": response.to_json(),
            }
        )
        df = pd.concat((df, series.to_frame().T), ignore_index=True)
        df.to_pickle(output_file)

# %%
