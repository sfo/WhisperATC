# %%
import io
import os

import dotenv
import pandas as pd
import scipy.io.wavfile as wavfile
from datasets import Audio, load_dataset
from deepgram import DeepgramClient, PrerecordedOptions
from tqdm import tqdm

# %%
dts = "jlvdoorn/atco2-asr"
spl = "validation"

print("Dataset: ", dts)
print("Split  : ", spl)

# %%
if (secrets_file := dotenv.find_dotenv("secrets.env", usecwd=True)) == "":
    raise FileNotFoundError("Could not find environment file holding the API key!")
else:
    dotenv.load_dotenv(secrets_file)

DG_KEY = os.environ["DEEPGRAM_API_KEY"]

# %%
dataset = load_dataset(dts)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

df = pd.DataFrame()
for s in tqdm(spl.split("+")):
    for i in tqdm(range(len(dataset[s]))):
        sample = dataset[s][i]["audio"]
        audio = sample["array"]
        sampling_rate = sample["sampling_rate"]
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        wavfile.write(byte_io, sampling_rate, audio)

        source = {
            "buffer": byte_io.read()
        }

        model = "nova-2-atc"
        deepgram = DeepgramClient(DG_KEY)
        options = PrerecordedOptions(
            model=model,
            smart_format=True,
        )
        response = deepgram.listen.rest.v("1").transcribe_file(
            source, options
        )

        series = pd.Series(
            {
                "split": s,
                "ref": dataset[s][i]["text"],
                "results": response.to_json()
            }
        )

        df = pd.concat((df, series.to_frame().T), ignore_index=True)

        df.to_pickle(
            dts.split("/")[-1]
            + "-"
            + spl
            + "-"
            + "deepgram"
            + "-"
            + model
            + ".pickle"
        )

        break

# %%
