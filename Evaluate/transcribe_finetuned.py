# %%
from model_loader import load_model
from Transcriptor import transcribe

# %%
dts = "jlvdoorn/atco2-asr"
mdl = "jlvdoorn/whisper-large-v3-atco2-asr"
spl = "train+validation"
wsp = "-".join(mdl.split("-")[1:3])

print("Dataset: ", dts)
print("Model  : ", mdl)
print("Split  : ", spl)
print("Whisper: ", wsp)

# %%
model = load_model(wsp, mdl)

# %%
transcribe(model, dts, spl, mdl)
