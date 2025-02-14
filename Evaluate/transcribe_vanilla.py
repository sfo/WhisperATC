# %%
from Transcriptor import Transcriptor, VanillaWhisperModel

# %%
dts = "jlvdoorn/atco2-asr"
mdl = "openai/whisper-large-v3"
spl = "validation"
wsp = "-".join(mdl.split("-")[1:])

print("Dataset: ", dts)
print("Model  : ", mdl)
print("Split  : ", spl)
print("Whisper: ", wsp)

# %%
model = VanillaWhisperModel(wsp)
transcriptor = Transcriptor(model, dts, spl)

# %%
clean, prmpt = transcriptor._transcribe("validation", 0)

# %%
print(clean.raw)
