"""V-JEPA 2 video backbone with LoRA on the last half of the encoder."""
import math

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

MODEL = 'facebook/vjepa2-vitl-fpc64-256'
REVISION = 'b3c1679b7c34d3255ef3547f27c7b226aefab26f'


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32.0):
        super().__init__()
        self.base_layer = base
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling


def load_vjepa2(frames=16, size=224, rank=16, alpha=32.0, seed=1701):
    """Pretrained encoder; trainable: LoRA on query/value of the last 12 layers,
    the patch projection and all LayerNorms."""
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
    config.crop_size = config.image_size = size
    config.frames_per_clip = frames
    config.in_chans = 3
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = AutoModel.from_pretrained(MODEL, revision=REVISION, config=config, attn_implementation='sdpa')
        model.predictor = None
        layers = model.encoder.layer
        for layer in layers:
            layer.attention.grid_size = size // 16
            layer.attention.grid_depth = frames // 2
        for p in model.parameters():
            p.requires_grad = False
        for layer in layers[len(layers) // 2:]:
            for name in ('query', 'value'):
                setattr(layer.attention, name, LoRALinear(getattr(layer.attention, name), rank, alpha))
    for p in model.encoder.embeddings.patch_embeddings.proj.parameters():
        p.requires_grad = True
    for m in model.encoder.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters(recurse=False):
                p.requires_grad = True
    return model


def video_features(model, video):
    """Mean of the last encoder tokens; video is (B, T, 3, H, W), ImageNet-normalized."""
    return model.encoder(pixel_values_videos=video).last_hidden_state.mean(1)
