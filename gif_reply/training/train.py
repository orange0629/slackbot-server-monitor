"""Training loop for the SigLIP gif-reply finetune.

Supports:
  * SigLIP sigmoid loss (default) or CLIP-style InfoNCE, with optional
    multi-positive masking (same gif in a batch = positive, not a negative).
  * GradCache two-pass accumulation, so the contrastive batch is
    batch_size * grad_accum instead of just batch_size.
  * Single-process two-stage runs: stage 1 warms heads on TGIF(+PEPE),
    stage 2 fine-tunes PEPE-only. One job, one `python -m` invocation.
  * Checkpoint selection on PEPE-dev retrieval (Recall@k / DCG@30), not loss.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .data import (
    DEFAULT_FRAME_CACHE_DIR, DEFAULT_TOKEN_CACHE_DIR,
    GifReplyDataset, collate, load_meta,
)
from .eval_dcg import evaluate_retrieval
from .losses import contrastive_loss, pos_mask_from_uids, tag_bce, total_loss
from .model import ModelConfig, SigLIPDualEncoder, freeze_bottom_half

logger = logging.getLogger(__name__)


def build_dataset(split, token_cache_dir, frame_cache_dir, n_tags, n_frames):
    examples_path = os.path.join(token_cache_dir, f"{split}.examples.pt")
    tokens_path = os.path.join(token_cache_dir, f"{split}.pt")
    if not os.path.exists(examples_path) or not os.path.exists(tokens_path):
        raise FileNotFoundError(
            f"Missing token cache for split={split}. Run "
            f"`python -m gif_reply.training.data build-token-cache --split {split}` first."
        )
    examples = torch.load(examples_path, weights_only=False)
    return GifReplyDataset(examples, tokens_path, frame_cache_dir, n_tags, n_frames=n_frames)


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_frac: float = 0.05) -> float:
    warmup = max(int(total_steps * warmup_frac), 1)
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def reapply_freeze(model: SigLIPDualEncoder, freeze_fraction: float):
    """Re-freeze towers for a new stage. Unfreezes everything first so a stage
    with a *lower* freeze_fraction than the previous one actually unfreezes."""
    for tower in (model.text_model, model.vision_model):
        for p in tower.parameters():
            p.requires_grad_(True)
    nf_t, nt_t = freeze_bottom_half(model.text_model, freeze_fraction)
    nf_v, nt_v = freeze_bottom_half(model.vision_model, freeze_fraction)
    logger.info(f"[freeze] text {nf_t}/{nt_t}  vision {nf_v}/{nt_v}  (frac={freeze_fraction})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--token-cache-dir", default=DEFAULT_TOKEN_CACHE_DIR)
    p.add_argument("--frame-cache-dir", default=DEFAULT_FRAME_CACHE_DIR)
    p.add_argument("--model-name", default="google/siglip-base-patch16-224")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--tag-loss-weight", type=float, default=0.1)
    p.add_argument("--freeze-fraction", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-dev-batches", type=int, default=80)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-frames", type=int, default=None,
                   help="frames/gif at train time (cache holds 4; fewer = bigger batch).")

    # Loss / batch.
    p.add_argument("--loss", choices=["sigmoid", "infonce"], default="sigmoid")
    p.add_argument("--multi-positive", action="store_true",
                   help="treat same-gif pairs in a batch as positives (sigmoid only).")
    p.add_argument("--grad-cache", action="store_true",
                   help="GradCache: contrastive batch = batch_size * grad_accum.")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--vision-no-grad", action="store_true",
                   help="encode frames under no_grad (only if vision fully frozen).")

    # Checkpoint selection.
    p.add_argument("--select-metric", default="recall@10",
                   choices=["recall@1", "recall@5", "recall@10", "dcg@30", "mrr", "contrastive"])
    p.add_argument("--reset-best", action="store_true",
                   help="don't inherit best metric from the resumed checkpoint.")
    p.add_argument("--retrieval-max-tweets", type=int, default=3000)

    # Two-stage (single process). Stage 1 warms heads on extra data; stage 2
    # is PEPE-only. Set --stage1-epochs 0 to skip stage 1 entirely.
    p.add_argument("--stage1-epochs", type=int, default=0)
    p.add_argument("--stage2-epochs", type=int, default=2)
    p.add_argument("--stage1-freeze-fraction", type=float, default=1.0,
                   help="stage-1 freeze (1.0 = head/projection warmup only).")
    # Repeatable: pass the trio once per extra augmentation source (e.g. TGIF
    # and Giphy). The Nth --stage1-extra-examples-jsonl pairs with the Nth
    # --stage1-extra-tokens-pt and --stage1-extra-frame-dir.
    p.add_argument("--stage1-extra-examples-jsonl", action="append", default=None)
    p.add_argument("--stage1-extra-tokens-pt", action="append", default=None)
    p.add_argument("--stage1-extra-frame-dir", action="append", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    meta = load_meta()
    n_tags = meta["n_labels"]

    cfg = ModelConfig(
        model_name=args.model_name, n_tags=n_tags, freeze_fraction=args.freeze_fraction,
        gradient_checkpointing=args.gradient_checkpointing, vision_no_grad=args.vision_no_grad,
    )
    model = SigLIPDualEncoder(cfg).to(device)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    global_step = 0
    best_metric = -float("inf")  # higher is better for retrieval metrics
    higher_is_better = args.select_metric != "contrastive"
    if not higher_is_better:
        best_metric = float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        state = ck["model"] if "model" in ck else ck
        # strict=False: a Phase-1 checkpoint has no logit_bias param.
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"resumed weights from {args.resume} "
                    f"(missing={list(missing)}, unexpected={list(unexpected)})")
        global_step = ck.get("global_step", 0)
        if not args.reset_best:
            best_metric = ck.get("best_metric", best_metric)

    dev_ds = build_dataset("dev", args.token_cache_dir, args.frame_cache_dir, n_tags, args.n_frames)
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(args.num_workers // 2, 1), pin_memory=True, collate_fn=collate,
    )
    log_path = os.path.join(args.output_dir, "train.jsonl")

    def log_event(d: dict):
        with open(log_path, "a") as f:
            f.write(json.dumps({"t": time.time(), **d}) + "\n")

    def save(name: str, extra: dict | None = None):
        out = os.path.join(args.output_dir, name)
        ck = {
            "model": model.state_dict(),
            "global_step": global_step,
            "best_metric": best_metric,
            "select_metric": args.select_metric,
            "config": cfg.__dict__,
        }
        if extra:
            ck.update(extra)
        tmp = out + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, out)

    @torch.no_grad()
    def dev_loss():
        model.eval()
        agg = {"loss": 0.0, "loss_contrastive": 0.0, "loss_tag_text": 0.0, "loss_tag_image": 0.0, "n": 0}
        for i, batch in enumerate(dev_loader):
            if batch is None:
                continue
            if i >= args.max_dev_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(batch["input_ids"], batch["attention_mask"], batch["frames"])
                pm = pos_mask_from_uids(batch["gif_uid"]) if args.multi_positive else None
                losses = total_loss(out, batch["tag_vec"], tag_weight=args.tag_loss_weight,
                                    loss_type=args.loss, pos_mask=pm)
            bs = batch["input_ids"].size(0)
            for k in ("loss", "loss_contrastive", "loss_tag_text", "loss_tag_image"):
                agg[k] += float(losses[k]) * bs
            agg["n"] += bs
        n = max(agg["n"], 1)
        return {k: agg[k] / n for k in ("loss", "loss_contrastive", "loss_tag_text", "loss_tag_image")}

    def selection_value():
        """Returns (value, dict) for the configured selection metric."""
        if args.select_metric == "contrastive":
            d = dev_loss()
            model.train()
            return d["loss_contrastive"], {"phase": "dev_loss", **d}
        m = evaluate_retrieval(model, "dev", args.frame_cache_dir, device,
                               max_tweets=args.retrieval_max_tweets, batch_size=args.batch_size)
        model.train()
        return m[args.select_metric], {"phase": "dev_retrieval", **m}

    def is_better(v: float) -> bool:
        return v > best_metric if higher_is_better else v < best_metric

    # ---- Stage plan ----
    # Zip the repeatable --stage1-extra-* trios into a list of (jsonl, tokens,
    # frame_dir). Each list is None (flag absent) or a same-length list.
    ejs = args.stage1_extra_examples_jsonl or []
    ets = args.stage1_extra_tokens_pt or []
    efs = args.stage1_extra_frame_dir or []
    if not (len(ejs) == len(ets) == len(efs)):
        raise ValueError(
            "each --stage1-extra-examples-jsonl needs a matching --stage1-extra-tokens-pt "
            f"and --stage1-extra-frame-dir (got {len(ejs)}/{len(ets)}/{len(efs)})"
        )
    extra_sources = list(zip(ejs, ets, efs))

    stages = []
    if args.stage1_epochs > 0:
        stages.append({
            "name": "stage1", "epochs": args.stage1_epochs,
            "freeze": args.stage1_freeze_fraction,
            "extra": extra_sources,
        })
    stages.append({
        "name": "stage2", "epochs": args.stage2_epochs,
        "freeze": args.freeze_fraction, "extra": [],
    })

    for st in stages:
        logger.info(f"==== {st['name']}: {st['epochs']} epoch(s), freeze={st['freeze']} ====")
        reapply_freeze(model, st["freeze"])

        train_ds = build_dataset("train", args.token_cache_dir, args.frame_cache_dir,
                                 n_tags, args.n_frames)
        for ej, et, ef in st["extra"]:
            if not (ej and et and ef):
                raise ValueError("stage1 extra needs jsonl + tokens-pt + frame-dir")
            with open(ej) as f:
                extra = [json.loads(line) for line in f]
            extra_ds = GifReplyDataset(extra, et, ef, n_tags, n_frames=args.n_frames)
            logger.info(f"[{st['name']}] +{len(extra_ds)} extra examples from {ej}")
            train_ds = ConcatDataset([train_ds, extra_ds])
        logger.info(f"[{st['name']}] train={len(train_ds)} dev={len(dev_ds)}")

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
            pin_memory=True, collate_fn=collate, drop_last=True,
        )
        # Fresh optimizer per stage: the trainable-param set changes with the
        # freeze fraction, and Adam moments shouldn't carry across regimes.
        groups = model.trainable_parameter_groups(base_lr=args.lr, head_lr=args.head_lr)
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        base_lrs = [g["lr"] for g in optimizer.param_groups]

        steps_per_epoch = len(train_loader) // max(args.grad_accum, 1)
        total_steps = max(steps_per_epoch * st["epochs"], 1)
        stage_step = 0

        optimizer.zero_grad(set_to_none=True)
        for epoch in range(st["epochs"]):
            model.train()
            t0 = time.time()
            running = {"loss": 0.0, "loss_contrastive": 0.0, "loss_tag_text": 0.0, "loss_tag_image": 0.0, "n": 0}
            micro: list[dict] = []

            def apply_step():
                nonlocal global_step, stage_step
                lr_mul = cosine_lr(stage_step, total_steps, base_lr=1.0)
                for g, base in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = base * lr_mul
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                stage_step += 1
                return lr_mul

            for batch in train_loader:
                if batch is None:
                    continue
                micro.append(batch)
                if len(micro) < args.grad_accum:
                    continue

                if args.grad_cache:
                    stats = _grad_cache_step(model, micro, device, amp_dtype, use_amp, args)
                else:
                    stats = _simple_accum_step(model, micro, device, amp_dtype, use_amp, args)
                micro = []
                lr_mul = apply_step()

                for k in ("loss", "loss_contrastive", "loss_tag_text", "loss_tag_image"):
                    running[k] += stats[k]
                running["n"] += 1

                if global_step % args.log_every == 0:
                    n = running["n"]
                    avg = {k: running[k] / n for k in
                           ("loss", "loss_contrastive", "loss_tag_text", "loss_tag_image")}
                    logger.info(
                        f"[{st['name']}] ep={epoch} step={global_step} lr_mul={lr_mul:.4f} "
                        f"loss={avg['loss']:.4f} contrastive={avg['loss_contrastive']:.4f} "
                        f"tag_t={avg['loss_tag_text']:.4f} tag_i={avg['loss_tag_image']:.4f}")
                    log_event({"phase": "train", "stage": st["name"], "epoch": epoch,
                               "step": global_step, "lr_mul": lr_mul, **avg})
                    running = {k: 0.0 for k in running}; running["n"] = 0

            # End of epoch. Persist latest.pt FIRST so a flaky selection eval
            # can never cost us a whole epoch of compute again.
            save("latest.pt")
            logger.info(f"[{st['name']}] epoch {epoch} done in {time.time()-t0:.1f}s; "
                        f"saved latest.pt")
            try:
                val, vlog = selection_value()
            except Exception:
                logger.exception(f"[{st['name']}] selection eval failed; "
                                 f"keeping latest.pt, skipping best update")
            else:
                logger.info(f"[{st['name']}] {args.select_metric}={val:.4f} "
                            f"(best={best_metric:.4f})")
                log_event({"stage": st["name"], "epoch": epoch, "step": global_step,
                           "select_metric": args.select_metric, "select_value": val, **vlog})
                if is_better(val):
                    best_metric = val
                    save("best.pt", extra={"select_value": val, **vlog})
                    logger.info(f"[{st['name']}] new best {args.select_metric}={val:.4f}")
                    save("latest.pt", extra={"select_value": val, **vlog})

    logger.info(f"done. best {args.select_metric}={best_metric:.4f}")


def _simple_accum_step(model, micro, device, amp_dtype, use_amp, args) -> dict:
    """Plain gradient accumulation: contrastive negatives limited to one
    micro-batch. Kept for --grad-cache off / debugging."""
    agg = {"loss": 0.0, "loss_contrastive": 0.0, "loss_tag_text": 0.0, "loss_tag_image": 0.0}
    for batch in micro:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch["input_ids"], batch["attention_mask"], batch["frames"])
            pm = pos_mask_from_uids(batch["gif_uid"]) if args.multi_positive else None
            losses = total_loss(out, batch["tag_vec"], tag_weight=args.tag_loss_weight,
                                loss_type=args.loss, pos_mask=pm)
            loss = losses["loss"] / len(micro)
        loss.backward()
        for k in ("loss", "loss_contrastive", "loss_tag_text", "loss_tag_image"):
            agg[k] += float(losses[k]) / len(micro)
    return agg


def _grad_cache_step(model, micro, device, amp_dtype, use_amp, args) -> dict:
    """GradCache (Gao et al. 2021): two passes so the contrastive loss sees
    the *whole* effective batch (batch_size * grad_accum) at the memory of one
    micro-batch.

      Pass A  no model grad: encode all micro-batches → embedding bank,
              compute contrastive loss over the full bank, backprop to get
              d(loss)/d(embeddings) (and logit_scale/bias grads directly).
      Pass B  re-encode each micro-batch with grad and backward the surrogate
              (embeds · cached_grad) + the per-example tag BCE.
    """
    # ---- Pass A: build the detached embedding bank ----
    text_bank, img_bank, uids = [], [], []
    for batch in micro:
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            t = model.encode_text(b["input_ids"], b["attention_mask"])
            im = model.encode_frames(b["frames"])
        text_bank.append(t.float())
        img_bank.append(im.float())
        uids.append(b["gif_uid"])

    T = torch.cat(text_bank).requires_grad_(True)
    I = torch.cat(img_bank).requires_grad_(True)
    pm = pos_mask_from_uids(torch.cat(uids)) if args.multi_positive else None
    l_c = contrastive_loss(
        {"text_embeds": T, "image_embeds": I,
         "logit_scale": model.logit_scale, "logit_bias": model.logit_bias},
        loss_type=args.loss, pos_mask=pm,
    )
    # Populates T.grad, I.grad and (directly) logit_scale/logit_bias .grad.
    l_c.backward()
    gT, gI = T.grad.detach(), I.grad.detach()

    # ---- Pass B: re-encode with grad, backward the cached surrogate + tags ----
    lc_val = l_c.detach().item()
    agg = {"loss": lc_val, "loss_contrastive": lc_val,
           "loss_tag_text": 0.0, "loss_tag_image": 0.0}
    off = 0
    for batch in micro:
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        n = b["input_ids"].size(0)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            t = model.encode_text(b["input_ids"], b["attention_mask"])
            im = model.encode_frames(b["frames"])
            text_logits = model.text_tag_head(t)
            image_logits = model.image_tag_head(im)
            l_tt, l_ti = tag_bce(text_logits, image_logits, b["tag_vec"])
            tag_term = args.tag_loss_weight * (l_tt + l_ti) / len(micro)
        # Contrastive surrogate: ∂surrogate/∂embeds == cached upstream grad,
        # so model weights get the exact full-batch contrastive gradient.
        surrogate = (t.float() * gT[off:off + n]).sum() + (im.float() * gI[off:off + n]).sum()
        (surrogate + tag_term).backward()
        agg["loss_tag_text"] += float(l_tt) / len(micro)
        agg["loss_tag_image"] += float(l_ti) / len(micro)
        off += n
    agg["loss"] += agg["loss_tag_text"] + agg["loss_tag_image"]
    return agg


if __name__ == "__main__":
    main()
