# %%
import pandas as pd

# %%
df = pd.read_pickle(
    "atco2-asr-atcosim-train+validation-whisper-large-v3-2024-07-22--03:47:28.pickle"
)

# %% [markdown]
# # Normalization

# %%
from Evaluate.Normalizer import filterAndNormalize

# %%
df["ref-norm"] = df.apply(lambda x: filterAndNormalize(x["ref"]), axis=1)
df["hyp-clean-norm"] = df.apply(lambda x: filterAndNormalize(x["hyp-clean"]), axis=1)
df["hyp-prmpt-norm"] = df.apply(lambda x: filterAndNormalize(x["hyp-prmpt"]), axis=1)

# %%
df[["ref-norm", "hyp-clean-norm"]]


# %%
mask_prmpt = df["ref-norm"] == df["hyp-prmpt-norm"]
mask_clean = df["ref-norm"] == df["hyp-clean-norm"]

# masks are equal, except for 1 record for which the clean model is more correct.

# %%
