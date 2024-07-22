# %%
import argparse

parser = argparse.ArgumentParser(description="Evaluate a transcription")
parser.add_argument("file", help="a dataframe in pickle format as output by transcription script", type=str)
args = parser.parse_args()

# %%
dts = 'jlvdoorn/atco2-asr-atcosim'
mdl = 'openai/whisper-large-v3'
spl = 'train+validation'
wsp = '-'.join(mdl.split('-')[1:])

print('Dataset: ', dts)
print('Model  : ', mdl)
print('Split  : ', spl)
print('Whisper: ', wsp)

# %%
import pandas as pd

# %%
df = pd.read_pickle(args.file)

# %% [markdown]
# # Normalization

# %%
from Normalizer import filterAndNormalize

# %%
df['ref-norm'] = df.apply(lambda x: filterAndNormalize(x['ref']), axis=1)
df['hyp-clean-norm'] = df.apply(lambda x: filterAndNormalize(x['hyp-clean']), axis=1)
df['hyp-prmpt-norm'] = df.apply(lambda x: filterAndNormalize(x['hyp-prmpt']), axis=1)

# %%
df.head()

# %% [markdown]
# # WER Calculation

# %%
import jiwer

# %%
def calcWER(df, spl):
    dff = df.loc[df['split'].isin(spl.split('+'))]
    wer_cln = jiwer.wer(list(dff['ref']), list(dff['hyp-clean']))
    wer_prm = jiwer.wer(list(dff['ref']), list(dff['hyp-prmpt']))
    wer_cln_nrm = jiwer.wer(list(dff['ref-norm']), list(dff['hyp-clean-norm']))
    wer_prm_nrm = jiwer.wer(list(dff['ref-norm']), list(dff['hyp-prmpt-norm']))

    print('clean        : {} %'.format(round(wer_cln*100,4)))
    print('prmpt        : {} %'.format(round(wer_prm*100,4)))
    print('clean-norm   : {} %'.format(round(wer_cln_nrm*100,4)))
    print('prmpt-norm   : {} %'.format(round(wer_prm_nrm*100,4)))

# %%
# Split Train+Validation
spl = 'train+validation'
wsp = '-'.join(mdl.split('-')[1:])

print('Dataset: ', dts)
print('Model  : ', mdl)
print('Split  : ', spl)
print('Whisper: ', wsp)

calcWER(df, spl)

# %%
# Split Validation
spl = 'validation'
wsp = '-'.join(mdl.split('-')[1:])

print('Dataset: ', dts)
print('Model  : ', mdl)
print('Split  : ', spl)
print('Whisper: ', wsp)

calcWER(df, spl)
