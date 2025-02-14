# %%
from Transcriptor import WhisperATCModel, transcribe

# %%
dts = "jlvdoorn/atco2-asr"
mdl = "jlvdoorn/whisper-large-v3-atco2-asr"
spl = "validation"
wsp = "-".join(mdl.split("-")[1:])

print("Dataset: ", dts)
print("Model  : ", mdl)
print("Split  : ", spl)
print("Whisper: ", wsp)

# %%
model = WhisperATCModel(wsp)

# %%
transcribe(model, dts, spl)
