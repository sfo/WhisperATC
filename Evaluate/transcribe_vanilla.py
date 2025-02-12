# %%
import whisper
from Transcriptor import transcribe

# %%
dts = "jlvdoorn/atco2-asr-atcosim"
mdl = "openai/whisper-large-v3"
spl = "train+validation"
wsp = "-".join(mdl.split("-")[1:])

print("Dataset: ", dts)
print("Model  : ", mdl)
print("Split  : ", spl)
print("Whisper: ", wsp)

# %%
model = whisper.load_model(wsp)

# %%
transcribe(model, dts, spl, mdl)
