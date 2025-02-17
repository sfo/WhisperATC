# %%
import os

import dotenv
from tqdm.auto import tqdm
from Transcriptor import AssemblyAIModel, Transcriptor

# %%
if (secrets_file := dotenv.find_dotenv("secrets.env", usecwd=True)) == "":
    raise FileNotFoundError("Could not find environment file holding the API key!")
else:
    dotenv.load_dotenv(secrets_file)

API_KEY = os.environ["ASSEMBLYAI_API_KEY"]

# %%
spl = "validation"
for dts in tqdm(
    (
        "jlvdoorn/atcosim",
        "jlvdoorn/atco2-asr",
    ),
    desc="iterating datasets ...",
):
    model = AssemblyAIModel(API_KEY)
    transcriptor = Transcriptor(model, dts, spl)
    transcriptor.transcribe()

# %%
