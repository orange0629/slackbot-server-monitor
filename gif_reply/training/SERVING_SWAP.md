# Serving swap: PEPE-v2 → live bot (operator runbook — NOT executed by the build)

These steps make the trained `pepe_v2/best.pt` + unified `index_pepe_v2/` the
bot's live GIF-reply backend. They are deliberately **not** applied by the
PEPE-v2 build. Do them only after reviewing `compare_pepe.py` results.

Prereqs (produced by the build): `models/pepe_v2/best.pt` and
`/shared/0/projects/gif-reply-slack-bot/index_pepe_v2/siglip_ft_embeddings.npy`
+ `index_metadata.jsonl`.

## 1. Add a `siglip_ft` serving encoder (`gif_reply/encoders.py`)

`load_encoder` (`encoders.py:133`) only knows `siglip`/`pepe`. The trained
model is `SigLIPDualEncoder` — pre-tokenized text, frame tensors — so it needs
a wrapper that satisfies the `TextEncoder` protocol (`encoders.py:12-15`):
`encode_text(str)->np.ndarray`, `encode_image(PIL.Image)->np.ndarray`, `.dim`.

Reference implementations to mirror:
- text/load path: `eval_dcg._load_encoder("siglip_ft", ...)` + `encode_tweets`
  (`eval_dcg.py:37-46`, `103-121`) — SigLIP processor, all-ones mask.
- single-PIL image path: `SiglipEncoder.encode_image` (`encoders.py:56-61`).
  Note: serving (`engine.py`, `giphy_refresh.py`) passes **PIL images**, not
  cached frame tensors, so the wrapper must implement `encode_image(PIL)`
  (encode the single frame through the vision tower) — `reindex.py` used
  `encode_frames` on cached tensors, a different entrypoint.

Sketch:

```python
class SiglipFtEncoder:
    def __init__(self, checkpoint_path, model_name="google/siglip-base-patch16-224"):
        import torch
        from transformers import AutoProcessor
        from .training.data import load_meta
        from .training.model import ModelConfig, SigLIPDualEncoder
        self._torch = torch
        meta = load_meta()
        cfg = ModelConfig(model_name=model_name, n_tags=meta["n_labels"])
        self.model = SigLIPDualEncoder(cfg).eval()
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"] if "model" in ck else ck)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.dim = 768
    def encode_text(self, text):
        with self._torch.no_grad():
            enc = self.processor(text=[text], return_tensors="pt",
                                 padding="max_length", truncation=True)
            ids = enc["input_ids"]
            attn = enc.get("attention_mask", self._torch.ones_like(ids))
            v = self.model.encode_text(ids, attn)[0]
        return (v / v.norm().clamp_min(1e-8)).cpu().numpy().astype("float32")
    def encode_image(self, image):
        # single PIL frame → (1,1,3,H,W) through the frame encoder
        with self._torch.no_grad():
            px = self.processor(images=image, return_tensors="pt")["pixel_values"]
            v = self.model.encode_frames(px.unsqueeze(1))[0]
        return (v / v.norm().clamp_min(1e-8)).cpu().numpy().astype("float32")
```

Then in `load_encoder` add:

```python
    if backend == "siglip_ft":
        return SiglipFtEncoder(checkpoint_path=kwargs["checkpoint_path"],
                               model_name=kwargs.get("model_name", "google/siglip-base-patch16-224"))
```

Verify the `encode_frames` input rank against `model.py` before relying on the
`encode_image` sketch (frame encoder expects `(B, n_frames, 3, H, W)`).

## 2. Config + main.py wiring

`config.py`:
- `GIF_REPLY_BACKEND = "siglip_ft"` (line 41)
- `GIF_REPLY_INDEX_DIR = "/shared/0/projects/gif-reply-slack-bot/index_pepe_v2"` (line 49)
- add `GIF_REPLY_FT_CHECKPOINT = "/shared/0/projects/gif-reply-slack-bot/models/pepe_v2/best.pt"`

`main.py` `_get_gif_reply_engine` (after the `elif "pepe"` at `main.py:583-584`):

```python
        elif GIF_REPLY_BACKEND == "siglip_ft":
            encoder_kwargs = {
                "checkpoint_path": GIF_REPLY_FT_CHECKPOINT,
                "model_name": GIF_REPLY_SIGLIP_MODEL,
            }
```

## 3. Index filename invariant (already satisfied)

`Index.load` reads `f"{backend}_embeddings.npy"` (`index.py:46`). The reindex
job wrote `siglip_ft_embeddings.npy` into `index_pepe_v2/`, so backend name
`siglip_ft` + that dir line up — no rename needed.

## 4. Giphy auto-refresh compatibility

`giphy_refresh.py` `--backend` choices are `["siglip","pepe"]`
(`giphy_refresh.py:626`); the scheduled `run_giphy_refresh` passes
`GIF_REPLY_BACKEND` (`main.py`). Add `"siglip_ft"` to that choices list so the
nightly refresh keeps working; it will append new Giphy GIFs using the new
encoder. Embedding dim must stay 768 to match the index — `_atomic_extend_index`
enforces a dim check (`giphy_refresh.py:473-476`).

## 5. Restart the bot

The engine is built once and cached in `_GIF_REPLY_ENGINE`
(`main.py:572-601`); Python does not hot-reload (see the
`project-bot-requires-restart` memory). Restart: `kill <parent pid>` then
`python main.py`. Sanity-check the startup log line
`GIF reply engine ready: backend=siglip_ft, index_size=<N>` and fire a test
message in `#gif-testing`.

## 6. Rollback

Revert the three `config.py` lines (back to `siglip` /
`/shared/0/projects/gif-reply-slack-bot/index`) and restart. The old `index/`
dir and the `siglip` backend are untouched by this swap, so rollback is
config-only.
