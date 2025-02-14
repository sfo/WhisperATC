# %%
import os

import dotenv
from Transcriptor import AssemblyAIModel, Transcriptor

# %%
dts = "jlvdoorn/atcosim"
spl = "validation"

print("Dataset: ", dts)
print("Split  : ", spl)

# %%
if (secrets_file := dotenv.find_dotenv("secrets.env", usecwd=True)) == "":
    raise FileNotFoundError("Could not find environment file holding the API key!")
else:
    dotenv.load_dotenv(secrets_file)

API_KEY = os.environ["ASSEMBLYAI_API_KEY"]

model = AssemblyAIModel(API_KEY)
transcriptor = Transcriptor(model, dts, spl)

# %%
clean, prmpt = transcriptor._transcribe("validation", 0)

# %%
clean.timestamps