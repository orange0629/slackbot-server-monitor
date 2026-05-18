"""Contrastive (InfoNCE or SigLIP sigmoid) + multitask BCE for the gif-reply finetune."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(text_embeds: torch.Tensor, image_embeds: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric softmax cross-entropy contrastive over an in-batch sample.

    Both embeds must be L2-normalized. logit_scale is ln(inverse-temp).
    Single-positive (diagonal) only — InfoNCE cannot represent multi-positive.
    """
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * (text_embeds @ image_embeds.t())  # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def pos_mask_from_uids(gif_uids: torch.Tensor) -> torch.Tensor:
    """(B,) gif uids → (B, B) bool positive mask (True where same gif)."""
    return gif_uids[:, None] == gif_uids[None, :]


def sigmoid_loss(
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    pos_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """SigLIP sigmoid pairwise loss (Zhai et al. 2023).

    Each (text_i, image_j) pair is scored independently — no batch-wide
    softmax — so it is numerically stable at huge batch and tolerates
    false negatives. With `pos_mask` it becomes multi-positive: any pair
    sharing a gif (same uid) is a positive, not just the diagonal.

    z = t·(x·y) + b ;  y_ij = +1 if positive else -1 ;  L = -mean Σ log σ(y_ij·z)
    Normalized by batch size (matches big_vision / open_clip).
    """
    b = text_embeds.size(0)
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * (text_embeds @ image_embeds.t()) + logit_bias  # (B, B)
    if pos_mask is None:
        pos_mask = torch.eye(b, dtype=torch.bool, device=logits.device)
    labels = torch.where(pos_mask, 1.0, -1.0)
    # -log σ(labels · logits), summed over the j axis, mean over the batch.
    return -F.logsigmoid(labels * logits).sum(dim=-1).mean()


def tag_bce(text_logits: torch.Tensor, image_logits: torch.Tensor, tag_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (L_tag_text, L_tag_image)."""
    lt = F.binary_cross_entropy_with_logits(text_logits, tag_vec)
    li = F.binary_cross_entropy_with_logits(image_logits, tag_vec)
    return lt, li


def contrastive_loss(
    out: dict,
    loss_type: str = "sigmoid",
    pos_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Just the contrastive term — used standalone by GradCache's pass A."""
    if loss_type == "sigmoid":
        return sigmoid_loss(
            out["text_embeds"], out["image_embeds"],
            out["logit_scale"], out["logit_bias"], pos_mask,
        )
    if loss_type == "infonce":
        return info_nce(out["text_embeds"], out["image_embeds"], out["logit_scale"])
    raise ValueError(f"unknown loss_type: {loss_type}")


def total_loss(
    out: dict,
    tag_vec: torch.Tensor,
    tag_weight: float = 0.1,
    loss_type: str = "sigmoid",
    pos_mask: torch.Tensor | None = None,
) -> dict:
    l_c = contrastive_loss(out, loss_type=loss_type, pos_mask=pos_mask)
    l_tt, l_ti = tag_bce(out["text_logits"], out["image_logits"], tag_vec)
    total = l_c + tag_weight * (l_tt + l_ti)
    return {
        "loss": total,
        "loss_contrastive": l_c.detach(),
        "loss_tag_text": l_tt.detach(),
        "loss_tag_image": l_ti.detach(),
    }
