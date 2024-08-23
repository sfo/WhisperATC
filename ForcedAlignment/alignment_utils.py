import matplotlib.pyplot as plt
import torch
import torchaudio.functional as F


def get_emission(waveform, bundle, device):
    model = bundle.get_model(with_star=False).to(device)
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
    return emission


def plot_emission(emission):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(emission.cpu().T)
    ax.set_title("Frame-wise class probabilities")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.tight_layout()


def align_emission(emission, tokens, device):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)
    alignments, scores = alignments[0], scores[0]
    scores = scores.exp()  # from log probs back to probs
    return alignments, scores


def align_tokens(emission, tokenized_transcript, device):
    aligned_tokens, alignment_scores = align_emission(
        emission, tokenized_transcript, device
    )
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
    return token_spans


def aggregate_words(tokens, transcript):
    lengths = [len(word) for word in transcript]
    assert len(tokens) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(tokens[i : i + l])
        i += l
    return ret


def seconds_to_timecode(seconds):
    return "{0:02d}:{1:02d}:{2:02d},{3:03d}".format(
        int(seconds // 3600),
        int(seconds // 60 % 60),
        int(seconds % 60),
        int((100.0 * seconds) % 100),
    )


def word_to_audacity_label(
    waveform,
    spans: list[dict[str, int]],
    num_frames: int,
    transcript: str,
    sample_rate: int,
):
    ratio = waveform.size(1) / num_frames
    start = int(ratio * spans[0]["start"]) / sample_rate
    end = int(ratio * spans[-1]["end"]) / sample_rate
    return f"{start:.3f}\t{end:.3f}\t{transcript}"


def plot_alignments(waveform, token_spans, emission, transcript, sample_rate):
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

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    axes[1].set_xlim([0, None])
    fig.tight_layout()
