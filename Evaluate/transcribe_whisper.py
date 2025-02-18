# %%
from tqdm.auto import tqdm
from Transcriptor import Transcriptor, VanillaWhisperModel, WhisperATCModel

# %%
spl = "validation"
wsp = "large-v3"
for dts in tqdm(
    (
        "jlvdoorn/atcosim",
        "jlvdoorn/atco2-asr",
    ),
    desc="iterating datasets ...",
):
    for model in tqdm(
        (
            VanillaWhisperModel(wsp),
            WhisperATCModel(f"{wsp}-{dts.split("/")[1]}"),
        ),
        desc="iterating models ...",
    ):
        transcriptor = Transcriptor(model, dts, spl)
        transcriptor.transcribe()

# %%
