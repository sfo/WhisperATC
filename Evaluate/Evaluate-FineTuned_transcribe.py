# %% [markdown]
# # Infering Original and HF Whisper

# %%
dts = 'jlvdoorn/atco2-asr'
mdl = 'jlvdoorn/whisper-large-v3-atco2-asr'
spl = 'train+validation'
wsp = '-'.join(mdl.split('-')[1:3])

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
import re
import whisper
import torch
import os
from safetensors.torch import load_file

def hf_to_whisper_states(text):
    """
    source: https://github.com/openai/whisper/discussions/830#discussioncomment-10026678
    """
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
    text = re.sub('proj_out.weight', 'decoder.token_embedding.weight', text)
    return text

if not os.path.exists(mdl.split('/')[-1]):
    os.system('git clone git@hf.co:'+mdl)
else:
    os.system('cd '+mdl.split('/')[-1]+' && git pull')
    os.system('cd '+mdl.split('/')[-1]+' && git lfs pull')
hf_state_dict = load_file('./'+mdl.split('/')[-1]+'/model.safetensors', device="cpu")

# Rename layers
for key in list(hf_state_dict.keys())[:]:
    new_key = hf_to_whisper_states(key)
    hf_state_dict[new_key] = hf_state_dict.pop(key)

# Init Whisper Model and replace model weights
model = whisper.load_model(wsp)
model.load_state_dict(hf_state_dict)

# %%
print('Starting inference...')
nato = "alpha,bravo,charlie,delta,echo,foxtrot,golf,hotel,india,juliett,kilo,lima,mike,november,oscar,papa,quebec,romeo,sierra,tango,uniform,victor,whiskey,xray,yankee,zulu"
terminology = "climb, climbing, descend, descending, passing, feet, knots, degrees, direct, maintain, identified, ILS, VFR, IFR, contact, frequency, turn, right, left, heading, altitude, flight, level, cleared, squawk, approach, runway, established, report, affirm, negative, wilco, roger, radio, radar"

for s in tqdm(spl.split('+')):
    print(' ')
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

