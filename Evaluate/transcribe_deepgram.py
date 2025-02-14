# %%
import os

import dotenv
from Transcriptor import DeepgramNova2ATC, DeepgramNova3, Transcriptor

# %%
if (secrets_file := dotenv.find_dotenv("secrets.env", usecwd=True)) == "":
    raise FileNotFoundError("Could not find environment file holding the API key!")
else:
    dotenv.load_dotenv(secrets_file)

DG_KEY = os.environ["DEEPGRAM_API_KEY"]

# %%
spl = "validation"
for dts in ("jlvdoorn/atcosim", "jlvdoorn/atci2-asr"):
    model = DeepgramNova2ATC(DG_KEY)
    transcriptor = Transcriptor(model, dts, spl)
    transcriptor.transcribe()

    model = DeepgramNova3(DG_KEY)
    transcriptor = Transcriptor(model, dts, spl)
    transcriptor.transcribe()
