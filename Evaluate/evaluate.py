# %%
import jiwer
import pandas as pd
from Normalizer import filterAndNormalize

# %%
dts = "jlvdoorn/atcosim"
mdl = "openai/whisper-large-v3"
spl = "validation"
wsp = "-".join(mdl.split("-")[1:])

print("Dataset: ", dts)
print("Model  : ", mdl)
print("Split  : ", spl)
print("Whisper: ", wsp)

# %%
df = pd.read_pickle(
    dts.split("/")[-1]
    + "-"
    + spl
    + "-"
    + mdl.split("/")[-1]
    + ".pickle"
)
# %% [markdown]
# # Normalization

# %%
df["ref-norm"] = df.apply(lambda x: filterAndNormalize(x["ref"]), axis=1)
df["hyp-clean-norm"] = df.apply(lambda x: filterAndNormalize(x["hyp-clean"]), axis=1)
df["hyp-prmpt-norm"] = df.apply(lambda x: filterAndNormalize(x["hyp-prmpt"]), axis=1)

# %% [markdown]
# # WER Calculation


# %%
def calcWER(df, spl):
    dff = df.loc[df["split"].isin(spl.split("+"))]
    wer_cln = jiwer.wer(list(dff["ref"]), list(dff["hyp-clean"]))
    wer_prm = jiwer.wer(list(dff["ref"]), list(dff["hyp-prmpt"]))
    wer_cln_nrm = jiwer.wer(list(dff["ref-norm"]), list(dff["hyp-clean-norm"]))
    wer_prm_nrm = jiwer.wer(list(dff["ref-norm"]), list(dff["hyp-prmpt-norm"]))

    print("clean        : {} %".format(round(wer_cln * 100, 4)))
    print("prmpt        : {} %".format(round(wer_prm * 100, 4)))
    print("clean-norm   : {} %".format(round(wer_cln_nrm * 100, 4)))
    print("prmpt-norm   : {} %".format(round(wer_prm_nrm * 100, 4)))


# %%
calcWER(df, spl)

# %%
