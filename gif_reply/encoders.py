"""Text encoders for GIF retrieval. CPU-only by design."""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class TextEncoder(Protocol):
    dim: int
    def encode_text(self, text: str) -> np.ndarray: ...
    def encode_image(self, image): ...  # PIL.Image -> np.ndarray


class SiglipEncoder:
    """Frozen HuggingFace SigLIP. Heavy imports are deferred to first use."""

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", cache_dir: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Probe dim by running a dummy text.
        with torch.no_grad():
            dummy = self.processor(text=["x"], return_tensors="pt", padding="max_length")
            out = self._unwrap(self.model.get_text_features(**dummy))
        self.dim = int(out.shape[-1])

    @staticmethod
    def _unwrap(out):
        # transformers 5.x can return a ModelOutput object instead of a tensor.
        if hasattr(out, "shape"):
            return out
        for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
            if hasattr(out, attr):
                v = getattr(out, attr)
                if v is not None:
                    return v
        raise TypeError(f"Unexpected encoder output type: {type(out)}")

    def encode_text(self, text: str) -> np.ndarray:
        with self._torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
            feats = self._unwrap(self.model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_image(self, image) -> np.ndarray:
        with self._torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            feats = self._unwrap(self.model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)


class PepeEncoder:
    """Loads the original PEPE checkpoint.

    The original model definition lives in
    /shared/2/projects/gif-reply/src/models/CLIP-variant-multitask/. We add that
    path to sys.path on construction and import the model class from it; we do
    not vendor it (yet) because it depends on transitive helpers in that tree.
    If the tree disappears we'll vendor.
    """

    SRC_PATH = "/shared/2/projects/gif-reply/src/models/CLIP-variant-multitask"

    def __init__(self, checkpoint_path: str):
        import os
        import sys
        import torch

        self._torch = torch
        if os.path.isdir(self.SRC_PATH) and self.SRC_PATH not in sys.path:
            sys.path.insert(0, self.SRC_PATH)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"PEPE checkpoint not found: {checkpoint_path}")

        # The original training tree exposes CLIPMultitaskModel under models.CLIP, but
        # importing it pulls in a project-local `opt` config. We try the simple path,
        # and if it fails we surface a clear error so the operator can vendor what's
        # needed. (The frozen SigLIP backend remains the recommended default.)
        ModelClass = None
        for module_path, class_name in (("models.CLIP", "CLIPMultitaskModel"),
                                        ("models.multimodal_clip", "MultimodalCLIP")):
            try:
                mod = __import__(module_path, fromlist=[class_name])
                ModelClass = getattr(mod, class_name)
                break
            except Exception:
                continue
        if ModelClass is None:
            raise ImportError(
                "Could not import the PEPE model class. The original training tree at "
                f"{self.SRC_PATH} requires a project-local `opt` config to import; "
                "vendor the minimum model definition into gif_reply/pepe_model.py to enable this backend."
            )

        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model = ModelClass()
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("PEPE load: %d missing, %d unexpected keys", len(missing), len(unexpected))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Try to read embedding dim from the model.
        self.dim = getattr(self.model, "embed_dim", 512)

    def encode_text(self, text: str) -> np.ndarray:
        with self._torch.no_grad():
            feats = self.model.encode_text([text])  # type: ignore[attr-defined]
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_image(self, image) -> np.ndarray:
        with self._torch.no_grad():
            feats = self.model.encode_image(image)  # type: ignore[attr-defined]
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)


def load_encoder(backend: str, **kwargs) -> TextEncoder:
    if backend == "siglip":
        return SiglipEncoder(
            model_name=kwargs.get("model_name", "google/siglip-base-patch16-224"),
            cache_dir=kwargs.get("cache_dir"),
        )
    if backend == "pepe":
        return PepeEncoder(checkpoint_path=kwargs["checkpoint_path"])
    raise ValueError(f"Unknown encoder backend: {backend}")
