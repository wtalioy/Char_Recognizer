import os
import torch
import hashlib
import warnings
import urllib
from tqdm import tqdm

_MODELS = {
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def load_state_dict(name: str, download_root: str = None):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found")

    with open(model_path, 'rb') as f:
        model = torch.jit.load(f, map_location="cpu").eval()

    v_state_dict = {k[len("visual."):]: v for k, v in model.state_dict().items() if k.startswith("visual.")}
    t_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("visual.")}

    return v_state_dict

def freeze(model):
    from model import LSTMCell
    for module in model.modules():
        if not isinstance(module, (LSTMCell, torch.nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad_(False)

def digitsvit(prompt_num=3):
    from model import DigitsViT
    model = DigitsViT(prompt_num=prompt_num)
    state_dict, _ = load_state_dict("ViT-L/14", download_root='.cache/clip')
    model.load_state_dict(state_dict, strict=False)
    freeze(model)
    return model