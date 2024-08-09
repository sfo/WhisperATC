# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.functional as F
from datasets import load_dataset
from scipy.io.wavfile import write

print(torch.__version__)
print(torchaudio.__version__)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
model_variant = "atcosim"
dts = f"jlvdoorn/{model_variant}"
spl = "train"
ds_audio = load_dataset(dts)[spl]

# %%
sample = ds_audio[42]
waveform = torch.atleast_2d(
    torch.tensor(sample["audio"]["array"], device=device, dtype=torch.float32)
)
sampling_rate = sample["audio"]["sampling_rate"]
TRANSCRIPT = sample["text"].strip().split()

# %%
bundle = torchaudio.pipelines.MMS_FA

model = bundle.get_model(with_star=False).to(device)
with torch.inference_mode():
    emission, _ = model(waveform.to(device))


# %%
def plot_emission(emission):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(emission.cpu().T)
    ax.set_title("Frame-wise class probabilities")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.tight_layout()


plot_emission(emission[0])

# %%
LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)

# %%
tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word]
" ".join(map(str, tokenized_transcript))


# %%
def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)
    alignments, scores = alignments[0], scores[0]
    scores = scores.exp()  # from log probs back to probs
    return alignments, scores


aligned_tokens, alignment_scores = align(emission, tokenized_transcript)
for i, (ali, score) in enumerate(zip(aligned_tokens, alignment_scores)):
    print(f"{i:3d}:\t{ali:2d} [{LABELS[ali]}], {score:.2f}")

# %%
token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
print("Token\tTime\tScore")
for s in token_spans:
    print(f"{LABELS[s.token]}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")


# %%
def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


word_spans = unflatten(token_spans, [len(word) for word in TRANSCRIPT])


# %%
def seconds_to_timecode(seconds):
    return "{0:02d}:{1:02d}:{2:02d},{3:03d}".format(
        int(seconds // 3600),
        int(seconds // 60 % 60),
        int(seconds % 60),
        int((100.0 * seconds) % 100),
    )


def preview_word(
    waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate
):
    ratio = waveform.size(1) / num_frames
    start = int(ratio * spans[0].start) / sample_rate
    end = int(ratio * spans[-1].end) / sample_rate
    return f"{start:.3f}\t{end:.3f}\t{transcript}"


num_frames = emission.size(1)

audio_file_path = Path(sample["audio"]["path"])
with open(audio_file_path.with_suffix(".txt"), "tw") as file:
    file.write(
        "\n".join(
            [
                preview_word(
                    waveform,
                    word_spans[i],
                    num_frames,
                    word,
                    sampling_rate,
                )
                for i, word in enumerate(TRANSCRIPT)
            ]
        )
    )

write(
    audio_file_path,
    sampling_rate,
    sample["audio"]["array"],
)


# %%
def plot_alignments(
    waveform, token_spans, emission, transcript, sample_rate=bundle.sample_rate
):
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
    axes[0].set_title("Emission")
    axes[0].set_xticks([])

    axes[1].specgram(waveform[0], Fs=sample_rate)
    for t_spans, chars in zip(token_spans, transcript):
        t0, t1 = t_spans[0].start + 0.1, t_spans[-1].end - 0.1
        axes[0].axvspan(
            t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white"
        )
        axes[1].axvspan(
            ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white"
        )
        axes[1].annotate(
            f"{_score(t_spans):.2f}",
            (ratio * t0, sample_rate * 0.51),
            annotation_clip=False,
        )

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    axes[1].set_xlim([0, None])
    fig.tight_layout()


plot_alignments(waveform, word_spans, emission, TRANSCRIPT)

# %%
