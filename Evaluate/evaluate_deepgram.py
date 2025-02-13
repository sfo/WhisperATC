# %%
import json

import jiwer
import pandas as pd
from Normalizer import filterAndNormalize


def parse_result(result):
    channels = json.loads(result)["results"]["channels"]
    assert len(channels) == 1
    alternatives = channels[0]["alternatives"]
    assert len(alternatives) == 1
    return alternatives[0]["transcript"]


# %%
dts = "jlvdoorn/atco2-asr"
spl = "validation"

print("Dataset: ", dts)
print("Split  : ", spl)

# %%
model = "nova-3"
# fmt: off
output_file = (dts.split("/")[-1]
    + "-"
    + spl
    + "-"
    + "deepgram"
    + "-"
    + model
    + ".pickle"
)
# fmt: on

df = pd.read_pickle(output_file).assign(
    transcript=lambda df: df["results"].transform(parse_result),
    ref_norm=lambda df: df.apply(lambda x: filterAndNormalize(x["ref"]), axis=1),
    transcript_norm=lambda df: df.apply(
        lambda x: filterAndNormalize(x["transcript"]), axis=1
    ),
)

# %% [markdown]
# # WER Calculation


# %%
def calcWER(df, spl):
    dff = df.loc[df["split"].isin(spl.split("+"))]
    wer_cln = jiwer.wer(list(dff["ref"]), list(dff["transcript"]))
    wer_cln_nrm = jiwer.wer(list(dff["ref_norm"]), list(dff["transcript_norm"]))

    print("clean        : {} %".format(round(wer_cln * 100, 4)))
    print("clean-norm   : {} %".format(round(wer_cln_nrm * 100, 4)))


# %%
calcWER(df, spl)

# %%
