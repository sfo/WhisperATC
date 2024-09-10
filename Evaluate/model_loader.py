import os
import re

import whisper
from safetensors.torch import load_file


def _hf_to_whisper_states(text):
    """
    source: https://github.com/openai/whisper/discussions/830#discussioncomment-10026678
    """
    text = re.sub(".layers.", ".blocks.", text)
    text = re.sub(".self_attn.", ".attn.", text)
    text = re.sub(".q_proj.", ".query.", text)
    text = re.sub(".k_proj.", ".key.", text)
    text = re.sub(".v_proj.", ".value.", text)
    text = re.sub(".out_proj.", ".out.", text)
    text = re.sub(".fc1.", ".mlp.0.", text)
    text = re.sub(".fc2.", ".mlp.2.", text)
    text = re.sub(".fc3.", ".mlp.3.", text)
    text = re.sub(".fc3.", ".mlp.3.", text)
    text = re.sub(".encoder_attn.", ".cross_attn.", text)
    text = re.sub(".cross_attn.ln.", ".cross_attn_ln.", text)
    text = re.sub(".embed_positions.weight", ".positional_embedding", text)
    text = re.sub(".embed_tokens.", ".token_embedding.", text)
    text = re.sub("model.", "", text)
    text = re.sub("attn.layer_norm.", "attn_ln.", text)
    text = re.sub(".final_layer_norm.", ".mlp_ln.", text)
    text = re.sub("encoder.layer_norm.", "encoder.ln_post.", text)
    text = re.sub("decoder.layer_norm.", "decoder.ln.", text)
    text = re.sub("proj_out.weight", "decoder.token_embedding.weight", text)
    return text


def _clone_model_repo(mdl) -> None:
    if not os.path.exists(mdl.split("/")[-1]):
        os.system("git clone https://hf.co/" + mdl)
    else:
        os.system("cd " + mdl.split("/")[-1] + " && git pull")
        os.system("cd " + mdl.split("/")[-1] + " && git lfs pull")


def load_model(whisper_base_model: str, mdl, device: str = "cpu"):
    _clone_model_repo(mdl)

    hf_state_dict = load_file(
        "./" + mdl.split("/")[-1] + "/model.safetensors", device=device
    )

    # Rename layers
    for key in list(hf_state_dict.keys())[:]:
        new_key = _hf_to_whisper_states(key)
        hf_state_dict[new_key] = hf_state_dict.pop(key)

    # Init Whisper Model and replace model weights
    model = whisper.load_model(whisper_base_model)
    model.load_state_dict(hf_state_dict)
    return model
