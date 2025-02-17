# %%
import os

import dotenv
from tqdm.auto import tqdm
from Transcriptor import DeepgramNova2ATC, DeepgramNova3, Transcriptor

# %%
if (secrets_file := dotenv.find_dotenv("secrets.env", usecwd=True)) == "":
    raise FileNotFoundError("Could not find environment file holding the API key!")
else:
    dotenv.load_dotenv(secrets_file)

DG_KEY = os.environ["DEEPGRAM_API_KEY"]

# %%
spl = "validation"
for dts in tqdm(
    (
        "jlvdoorn/atcosim",
        "jlvdoorn/atco2-asr",
    ),
    desc="iterating datasets ...",
):
    for model in tqdm(
        (
            DeepgramNova2ATC,
            DeepgramNova3,
        ),
        desc="iterating models ...",
        leave=False,
    ):
        transcriptor = Transcriptor(model(DG_KEY), dts, spl)
        transcriptor.transcribe()

# %%
