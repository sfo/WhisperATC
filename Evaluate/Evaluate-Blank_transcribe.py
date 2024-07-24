# %% [markdown]
# # Infering Original and HF Whisper

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
from datasets import load_dataset, Audio
dataset = load_dataset(dts)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset

# %%
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# %%
df = pd.DataFrame()

# %% [markdown]
# # Infering Original Whisper with HF Dataset

# %%
import whisper
model = whisper.load_model('-'.join(mdl.split('-')[1:]))

# %%
print('Starting inference...')
nato = "alpha,bravo,charlie,delta,echo,foxtrot,golf,hotel,india,juliett,kilo,lima,mike,november,oscar,papa,quebec,romeo,sierra,tango,uniform,victor,whiskey,xray,yankee,zulu"
terminology = "climb, climbing, descend, descending, passing, feet, knots, degrees, direct, maintain, identified, ILS, VFR, IFR, contact, frequency, turn, right, left, heading, altitude, flight, level, cleared, squawk, approach, runway, established, report, affirm, negative, wilco, roger, radio, radar"

for s in spl.split('+'):
    for i in tqdm(range(len(dataset[s]))):
        audio = dataset[s][i]['audio']['array']
        audio = np.float32(whisper.pad_or_trim(audio))

        try:
            prompt = 'Air Traffic Control Communications ' + dataset[s][i]['info'].replace('\n', ' ') + ' ' + nato.replace(',',' ') + ' ' + terminology.replace(',',' ')
        except:
            inf = ''
            prompt = 'Air Traffic Control Communications ' + nato.replace(',',' ') + ' ' + terminology.replace(',',' ')

        options = dict(language='en', prompt=prompt, fp16=False, word_timestamps=True)
        res_prmpt = whisper.transcribe(model, audio, **options)

        options = dict(language='en', fp16=False, word_timestamps=True)
        res_clean = whisper.transcribe(model, audio, **options)

        series = pd.Series({
            'split': s,
            'hyp-prmpt': res_prmpt['text'],
            'hyp-clean': res_clean['text'],
            'ref': dataset[s][i]['text'],
            'words-prmpt': res_prmpt['segments'],
            'words-clean': res_clean['segments'],
        })
        df = pd.concat((df, series.to_frame().T), ignore_index=True)

df.to_pickle(dts.split('/')[-1]+'-'+spl+'-'+mdl.split('/')[-1]+'-'+datetime.today().strftime('%Y-%m-%d--%H:%M:%S')+'.pickle')

# %%
df
