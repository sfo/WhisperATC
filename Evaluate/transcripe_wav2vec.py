# %%
from datasets import load_dataset, Audio
import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
import torchaudio.functional as F
from itertools import groupby

# %%
USE_LM = False
DATASET_ID = "Jzuluaga/uwb_atcc"
MODEL_ID = "Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-uwb-atcc"

# 1. Load the dataset
# we only load the 'test' partition, however, if you want to load the 'train' partition, you can change it accordingly
uwb_atcc_corpus = load_dataset(DATASET_ID)
uwb_atcc_corpus_test = uwb_atcc_corpus["test"]

# %%
# 2. Load the model
model = AutoModelForCTC.from_pretrained(MODEL_ID)

# %%
# 3. Load the processors, we offer support with LM, which should yield better resutls
if USE_LM:
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(MODEL_ID)
else:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

# %%
# 4. Format the test sample
sample = next(iter(uwb_atcc_corpus_test))
file_sampling_rate = sample['audio']['sampling_rate']
# resample if neccessary
if file_sampling_rate != 16000:
    resampled_audio = F.resample(torch.tensor(sample["audio"]["array"]), file_sampling_rate, 16000).numpy()
    file_sampling_rate = 16000
else:
    resampled_audio = torch.tensor(sample["audio"]["array"]).numpy()

# %%
# 5. Run the forward pass in the model
with torch.no_grad():
    logits = model(input_values).logits

# %%
# get the transcription with processor
if USE_LM:
    transcription = processor.batch_decode(logits.numpy()).text
else:
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]
# print the output
print(transcription)

# %%
# retrieve word-level timestamps
# source: https://github.com/huggingface/transformers/issues/11307#issuecomment-867648870

##############
# this is where the logic starts to get the start and end timestamp for each word
##############
words = [w for w in transcription.split(' ') if len(w) > 0]
predicted_ids = pred_ids[0].tolist()
duration_sec = input_values.shape[1] / file_sampling_rate

ids_w_time = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in enumerate(predicted_ids)]
# remove entries which are just "padding" (i.e. no characers are recognized)
ids_w_time = [i for i in ids_w_time if i[1] != processor.tokenizer.pad_token_id]

# now split the ids into groups of ids where each group represents a word
split_ids_w_time = [list(group) for k, group
                    in groupby(ids_w_time, lambda x: x[1] == processor.tokenizer.word_delimiter_token_id)
                    if not k]

assert len(split_ids_w_time) == len(words)  # make sure that there are the same number of id-groups as words. Otherwise something is wrong

word_start_times = []
word_end_times = []
for cur_ids_w_time, cur_word in zip(split_ids_w_time, words):
    _times = [_time for _time, _id in cur_ids_w_time]
    word_start_times.append(min(_times))
    word_end_times.append(max(_times))

words, word_start_times, word_end_times

# %%
