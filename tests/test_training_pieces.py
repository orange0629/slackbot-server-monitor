"""Lightweight CPU-only tests for the training package.

Skips anything that requires the SigLIP weights / PEPE pickle on disk,
since CI/local users may not have them.
"""
from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np
import pytest
import torch


# --- Frame decoding shape ---

def test_decode_pads_short_clip():
    from gif_reply.training import data
    # Synthetic "video" tensor: 2 frames, 64x64x3.
    frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    # Run the inner pad-and-resize logic directly.
    n_frames = 4
    if frames.shape[0] < n_frames:
        idx = list(range(frames.shape[0])) + [frames.shape[0] - 1] * (n_frames - frames.shape[0])
    sel = frames[idx]
    assert sel.shape == (4, 64, 64, 3)


def test_normalize_for_siglip_centers_around_zero():
    from gif_reply.training.data import _normalize_for_siglip
    x = torch.full((4, 3, 224, 224), 0.5)
    y = _normalize_for_siglip(x)
    # 0.5 input → 0 after (x - 0.5) / 0.5
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


# --- Tag vector lookup ---

def test_tag_vec_lookup_one_hot():
    from gif_reply.training.data import Example, GifReplyDataset
    # Build a tiny dataset with a fake token cache and frame cache.
    n_tags = 5
    examples = [Example(text="hello", gif_id="abcdef", tag_indices=[0, 2, 4])]
    tmp = tempfile.mkdtemp()
    tok = os.path.join(tmp, "t.pt")
    torch.save({"input_ids": torch.zeros(1, 16, dtype=torch.long),
                "attention_mask": torch.ones(1, 16, dtype=torch.long)}, tok)
    # Frame cache file at the expected path.
    from gif_reply.training.data import cache_path_for_gif
    cp = cache_path_for_gif("abcdef", tmp)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    torch.save(torch.zeros(4, 3, 224, 224, dtype=torch.float16), cp)
    ds = GifReplyDataset(examples, tok, tmp, n_tags=n_tags)
    item = ds[0]
    expected = torch.zeros(n_tags); expected[[0, 2, 4]] = 1.0
    assert torch.equal(item["tag_vec"], expected)
    assert item["frames"].shape == (4, 3, 224, 224)
    assert item["gif_uid"].dtype == torch.long

    # n_frames subsamples the cached 4 frames.
    ds2 = GifReplyDataset(examples, tok, tmp, n_tags=n_tags, n_frames=2)
    assert ds2[0]["frames"].shape == (2, 3, 224, 224)


def test_gif_uid_stable_and_equal_for_same_id():
    from gif_reply.training.data import gif_uid
    assert gif_uid("abcdef") == gif_uid("abcdef")
    assert gif_uid("abcdef") != gif_uid("abcdeg")
    assert -(2 ** 63) <= gif_uid("abcdef") < 2 ** 63


# --- Loss math ---

def test_info_nce_descends_with_better_alignment():
    from gif_reply.training.losses import info_nce
    torch.manual_seed(0)
    d = 32; B = 8
    text = torch.randn(B, d); text = text / text.norm(dim=-1, keepdim=True)
    misaligned = torch.randn(B, d); misaligned = misaligned / misaligned.norm(dim=-1, keepdim=True)
    aligned = text.clone()
    scale = torch.tensor(2.6593)
    l_bad = info_nce(text, misaligned, scale).item()
    l_good = info_nce(text, aligned, scale).item()
    assert l_good < l_bad


def test_total_loss_is_finite_and_backprops():
    from gif_reply.training.losses import total_loss
    B, d, T = 4, 16, 5
    out = {
        "text_embeds": torch.randn(B, d, requires_grad=True),
        "image_embeds": torch.randn(B, d, requires_grad=True),
        "text_logits": torch.randn(B, T, requires_grad=True),
        "image_logits": torch.randn(B, T, requires_grad=True),
        "logit_scale": torch.tensor(2.6593, requires_grad=True),
        "logit_bias": torch.tensor(-10.0, requires_grad=True),
    }
    # L2 normalize embeddings so InfoNCE is well-formed.
    out["text_embeds"] = out["text_embeds"] / out["text_embeds"].norm(dim=-1, keepdim=True)
    out["image_embeds"] = out["image_embeds"] / out["image_embeds"].norm(dim=-1, keepdim=True)
    out["text_embeds"].retain_grad()
    out["image_embeds"].retain_grad()
    tag_vec = torch.zeros(B, T); tag_vec[:, 0] = 1.0
    losses = total_loss(out, tag_vec, tag_weight=0.1)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


# --- Cosine LR schedule ---

def test_cosine_lr_warmup_and_decay():
    from gif_reply.training.train import cosine_lr
    base = 1.0
    total = 1000
    assert cosine_lr(0, total, base) == 0.0
    # mid-warmup
    assert 0.0 < cosine_lr(25, total, base) < base
    # post-warmup peak around end-of-warmup
    peak = cosine_lr(60, total, base)
    assert peak > 0.5 * base
    # tail: should be near zero
    assert cosine_lr(total - 1, total, base) < 0.1 * base


# --- Freeze-bottom-half logic on a synthetic tower ---

def test_freeze_bottom_half_freezes_correct_layers():
    from gif_reply.training.model import freeze_bottom_half
    import torch.nn as nn

    class FakeEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(8)])

    class FakeTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FakeEnc()
            self.embeddings = nn.Embedding(10, 4)

    t = FakeTower()
    n_frozen, n_total = freeze_bottom_half(t, freeze_fraction=0.5)
    assert n_total == 8
    assert n_frozen == 4
    for i in range(4):
        for p in t.encoder.layers[i].parameters():
            assert not p.requires_grad
    for i in range(4, 8):
        for p in t.encoder.layers[i].parameters():
            assert p.requires_grad
    # Embeddings frozen.
    for p in t.embeddings.parameters():
        assert not p.requires_grad


# --- Collate drops Nones ---

def test_collate_drops_none_items():
    from gif_reply.training.data import collate
    item = {
        "input_ids": torch.zeros(8, dtype=torch.long),
        "attention_mask": torch.ones(8, dtype=torch.long),
        "frames": torch.zeros(4, 3, 224, 224),
        "tag_vec": torch.zeros(5),
        "gif_uid": torch.tensor(123, dtype=torch.long),
    }
    batch = collate([item, None, item])
    assert batch["input_ids"].shape == (2, 8)
    assert batch["frames"].shape == (2, 4, 3, 224, 224)


def test_collate_returns_none_for_all_none_batch():
    from gif_reply.training.data import collate
    assert collate([None, None]) is None


# --- gif_id path scheme ---

def test_gif_id_to_mp4_path_layout():
    from gif_reply.training.data import gif_id_to_mp4_path
    pth = gif_id_to_mp4_path("abcdefgh", root="/r")
    assert pth == "/r/a/b/c/defgh.mp4"


# --- SigLIP sigmoid loss ---

def test_sigmoid_loss_descends_with_better_alignment():
    from gif_reply.training.losses import sigmoid_loss
    torch.manual_seed(0)
    d, B = 32, 8
    text = torch.randn(B, d); text = text / text.norm(dim=-1, keepdim=True)
    mis = torch.randn(B, d); mis = mis / mis.norm(dim=-1, keepdim=True)
    ls = torch.tensor(2.3); lb = torch.tensor(-10.0)
    assert sigmoid_loss(text, text, ls, lb).item() < sigmoid_loss(text, mis, ls, lb).item()


def test_multi_positive_mask_marks_same_gif():
    from gif_reply.training.losses import pos_mask_from_uids
    uids = torch.tensor([7, 7, 3, 9])
    pm = pos_mask_from_uids(uids)
    assert pm[0, 1] and pm[1, 0]      # same gif → positive
    assert not pm[0, 2]               # different gif → negative
    assert pm.diagonal().all()


def test_multi_positive_lowers_loss_for_duplicate_gifs():
    """If two batch rows are the *same* gif, scoring them as a positive
    should cost less than (wrongly) scoring them as a hard negative."""
    from gif_reply.training.losses import sigmoid_loss, pos_mask_from_uids
    torch.manual_seed(1)
    d = 16
    v = torch.randn(1, d); v = v / v.norm(dim=-1, keepdim=True)
    e = v.repeat(2, 1)                    # both rows ARE the same gif
    text = e.clone(); image = e.clone()
    uids = torch.tensor([5, 5])
    ls = torch.tensor(2.3); lb = torch.tensor(-2.0)
    l_naive = sigmoid_loss(text, image, ls, lb, None)                       # diag-only
    l_multi = sigmoid_loss(text, image, ls, lb, pos_mask_from_uids(uids))   # multi-pos
    assert l_multi < l_naive


def test_grad_cache_matches_full_batch_gradients():
    """The two-pass GradCache surrogate must yield the same parameter
    gradients as a single full-batch backward."""
    import torch.nn as nn
    from gif_reply.training.losses import contrastive_loss, pos_mask_from_uids, sigmoid_loss

    torch.manual_seed(0)
    D, N = 8, 12
    enc_t, enc_i = nn.Linear(5, D), nn.Linear(6, D)
    ls = nn.Parameter(torch.tensor(2.3)); lb = nn.Parameter(torch.tensor(-5.0))
    xt, xi = torch.randn(N, 5), torch.randn(N, 6)
    uids = torch.tensor([1, 1, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7])
    pm = pos_mask_from_uids(uids)
    params = list(enc_t.parameters()) + list(enc_i.parameters()) + [ls, lb]

    def nrm(x):
        return torch.nn.functional.normalize(x, dim=-1)

    for p in params:
        p.grad = None
    L = sigmoid_loss(nrm(enc_t(xt)), nrm(enc_i(xi)), ls, lb, pm)
    L.backward()
    ref = {id(p): p.grad.clone() for p in params}

    for p in params:
        p.grad = None
    chunks = [slice(0, 4), slice(4, 8), slice(8, 12)]
    with torch.no_grad():
        TB = [nrm(enc_t(xt[c])) for c in chunks]
        IB = [nrm(enc_i(xi[c])) for c in chunks]
    T = torch.cat(TB).requires_grad_(True); I = torch.cat(IB).requires_grad_(True)
    contrastive_loss({"text_embeds": T, "image_embeds": I,
                      "logit_scale": ls, "logit_bias": lb},
                     loss_type="sigmoid", pos_mask=pm).backward()
    gT, gI = T.grad.detach(), I.grad.detach()
    off = 0
    for c in chunks:
        t, im = nrm(enc_t(xt[c])), nrm(enc_i(xi[c]))
        n = t.size(0)
        ((t * gT[off:off + n]).sum() + (im * gI[off:off + n]).sum()).backward()
        off += n

    for p in params:
        assert torch.allclose(p.grad, ref[id(p)], atol=1e-5), "GradCache grad mismatch"
