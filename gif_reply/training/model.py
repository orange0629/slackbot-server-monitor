"""SigLIP dual encoder with multitask tag heads and bottom-half freezing."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name: str = "google/siglip-base-patch16-224"
    n_tags: int = 241
    n_frames: int = 4
    freeze_fraction: float = 0.5  # freeze the bottom 50% of transformer layers
    gradient_checkpointing: bool = False  # trade compute for VRAM (bigger batch)
    vision_no_grad: bool = False  # encode frames under no_grad (vision fully frozen)


def _unwrap(out):
    """Same trick as gif_reply/encoders.py:38 — works on transformers 5.x."""
    if hasattr(out, "shape"):
        return out
    for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(out, attr):
            v = getattr(out, attr)
            if v is not None:
                return v
    raise TypeError(f"Unexpected encoder output type: {type(out)}")


def _list_layers(module: nn.Module) -> list[nn.Module] | None:
    """Find the encoder.layers ModuleList inside a SigLIP tower."""
    # SigLIP text/vision tower has .encoder.layers
    enc = getattr(module, "encoder", None)
    if enc is None:
        return None
    layers = getattr(enc, "layers", None)
    if layers is None:
        return None
    return list(layers)


def freeze_bottom_half(tower: nn.Module, freeze_fraction: float = 0.5) -> tuple[int, int]:
    """Freeze bottom `freeze_fraction` of transformer layers + token/patch embeddings.

    Returns (n_frozen_layers, n_total_layers) for logging.
    """
    layers = _list_layers(tower)
    if not layers:
        # Conservative: freeze nothing if we can't find layers.
        logger.warning("could not find encoder.layers on tower; nothing frozen")
        return 0, 0
    n_total = len(layers)
    n_frozen = int(round(n_total * freeze_fraction))
    for i in range(n_frozen):
        for p in layers[i].parameters():
            p.requires_grad_(False)
    # Also freeze the input embeddings (token/patch).
    for name in ("embeddings", "patch_embedding"):
        emb = getattr(tower, name, None)
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad_(False)
    return n_frozen, n_total


class SigLIPDualEncoder(nn.Module):
    """Dual SigLIP encoder with multitask tag heads.

    Forward inputs:
        input_ids, attention_mask: (B, L)
        frames: (B, N_FRAMES, 3, H, W)

    Forward outputs:
        text_embeds, image_embeds: (B, D) — L2-normalized
        text_logits, image_logits: (B, N_TAGS) — pre-sigmoid
        logit_scale: scalar
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from transformers import AutoModel, AutoProcessor  # heavy import deferred
        self.cfg = cfg
        base = AutoModel.from_pretrained(cfg.model_name)
        # SiglipModel has .text_model and .vision_model + .text_projection / .visual_projection
        self.text_model = base.text_model
        self.vision_model = base.vision_model
        # SigLIP's projection layers are exposed as `text_projection` (Linear)
        # and `visual_projection` is internal — we use the outputs of
        # get_text_features / get_image_features through unwrap.
        self._base = base
        self.processor = AutoProcessor.from_pretrained(cfg.model_name)

        # Probe embedding dim with a dummy forward.
        with torch.no_grad():
            dummy_in = self.processor(text=["x"], return_tensors="pt", padding="max_length", truncation=True)
            t = _unwrap(base.get_text_features(**dummy_in))
        d = int(t.shape[-1])
        self.embed_dim = d

        # Tag heads.
        self.text_tag_head = nn.Linear(d, cfg.n_tags)
        self.image_tag_head = nn.Linear(d, cfg.n_tags)

        # Learnable temperature + bias. SigLIP's sigmoid loss needs both; the
        # paper inits t' = log(10) (temperature) and b = -10 (bias). InfoNCE
        # only uses logit_scale and ignores logit_bias.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

        # Apply bottom-half freezing.
        nf_t, nt_t = freeze_bottom_half(self.text_model, cfg.freeze_fraction)
        nf_v, nt_v = freeze_bottom_half(self.vision_model, cfg.freeze_fraction)
        logger.info(f"frozen text layers: {nf_t}/{nt_t}; vision layers: {nf_v}/{nt_v}")

        if cfg.gradient_checkpointing:
            for tower in (self.text_model, self.vision_model):
                if hasattr(tower, "gradient_checkpointing_enable"):
                    # use_reentrant=False is required for frozen-bottom + checkpointing.
                    tower.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
            logger.info("gradient checkpointing enabled on both towers")

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        feats = _unwrap(self._base.get_text_features(input_ids=input_ids, attention_mask=attention_mask))
        return F.normalize(feats, dim=-1)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, N, 3, H, W) → (B, D), L2-normalized.

        When cfg.vision_no_grad is set the vision tower is treated as a fixed
        feature extractor (no autograd graph, big memory win). Only do this
        when the vision tower is fully frozen, else its grads are lost.
        """
        b, n, c, h, w = frames.shape
        flat = frames.reshape(b * n, c, h, w)
        ctx = torch.no_grad() if self.cfg.vision_no_grad else torch.enable_grad()
        with ctx:
            feats = _unwrap(self._base.get_image_features(pixel_values=flat))  # (B*N, D)
        feats = feats.reshape(b, n, -1).mean(dim=1)
        return F.normalize(feats, dim=-1)

    def forward(self, input_ids, attention_mask, frames):
        text_embeds = self.encode_text(input_ids, attention_mask)
        image_embeds = self.encode_frames(frames)
        text_logits = self.text_tag_head(text_embeds)
        image_logits = self.image_tag_head(image_embeds)
        return {
            "text_embeds": text_embeds,
            "image_embeds": image_embeds,
            "text_logits": text_logits,
            "image_logits": image_logits,
            "logit_scale": self.logit_scale,
            "logit_bias": self.logit_bias,
        }

    def trainable_parameter_groups(self, base_lr: float, head_lr: float):
        """Two param groups: tower params at base_lr, heads + projections at head_lr."""
        head_params, tower_params = [], []
        head_modules = (self.text_tag_head, self.image_tag_head)
        head_param_ids = {id(p) for m in head_modules for p in m.parameters()}
        head_param_ids.add(id(self.logit_scale))
        head_param_ids.add(id(self.logit_bias))
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in head_param_ids:
                head_params.append(p)
            else:
                tower_params.append(p)
        return [
            {"params": tower_params, "lr": base_lr},
            {"params": head_params, "lr": head_lr},
        ]
