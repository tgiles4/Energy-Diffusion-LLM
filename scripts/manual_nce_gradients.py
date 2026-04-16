#!/usr/bin/env python3
"""Two manual NCE steps on dummy text: loss, backward, optimizer, grad norms.

Run from repo root (needs HF checkpoint + CUDA for this codebase's autocast paths):

  python scripts/manual_nce_gradients.py

Optional:

  set MDLM_CKPT=kuleshov-group/mdlm-owt
  python scripts/manual_nce_gradients.py --length 128 --steps 2
"""
from __future__ import annotations

import sys
import types

# Some Windows hosts block Pillow's native DLL (policy / AV). Torchmetrics pulls in
# torchvision → PIL at import time; stub PIL so Lightning/torchmetrics can import.
def _install_pil_stub_if_broken() -> None:
  try:
    import PIL.Image  # noqa: F401
  except Exception:
    img_mod = types.ModuleType("PIL.Image")

    class Image:
      pass

    img_mod.Image = Image
    pil_pkg = types.ModuleType("PIL")
    sys.modules["PIL"] = pil_pkg
    sys.modules["PIL.Image"] = img_mod
    pil_pkg.Image = Image


_install_pil_stub_if_broken()

import argparse
import itertools
import os
import sys
from pathlib import Path

import lightning as L
import omegaconf
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Repo imports (same setup as main.py)
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))

import dataloader  # noqa: E402
import diffusion  # noqa: E402

_orig_torch_load = torch.load


def _torch_load_allow_trusted(*args, **kwargs):
  if "weights_only" not in kwargs:
    kwargs["weights_only"] = False
  return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_allow_trusted

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver(
    "div_up", lambda x, y: (x + y - 1) // y
)


def _build_config(overrides: list[str]) -> omegaconf.DictConfig:
  cfg_dir = str(_REPO / "configs")
  with initialize_config_dir(version_base=None, config_dir=cfg_dir):
    return compose(config_name="config", overrides=overrides)


def _grad_norm(params) -> float:
  sq = 0.0
  for p in params:
    if p.grad is not None:
      sq += float(p.grad.data.float().pow(2).sum())
  return sq ** 0.5


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--checkpoint",
      default=os.environ.get("MDLM_CKPT", "kuleshov-group/mdlm-owt"),
      help="HF model id or local path (same as eval.checkpoint_path).",
  )
  parser.add_argument("--length", type=int, default=128, help="Sequence length.")
  parser.add_argument("--steps", type=int, default=2, help="Manual optimizer steps.")
  parser.add_argument("--lr", type=float, default=3e-4)
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is required: ebm_forward uses torch.cuda.amp.autocast in this repo."
    )

  os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

  overrides = [
      "+path=.",
      "train_mdlm_only=false",
      "ebm_backbone=hf_dit",
      "ebm_readout=token_additive",
      f"eval.checkpoint_path={args.checkpoint}",
      f"model.length={args.length}",
      "trainer.devices=1",
      "trainer.num_nodes=1",
      "loader.global_batch_size=2",
      "loader.batch_size=2",
      "loader.eval_global_batch_size=2",
      "loader.num_workers=0",
      "training.ema=0",
  ]
  config = _build_config(overrides)
  OmegaConf.resolve(config)

  L.seed_everything(int(config.seed))

  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.EBM(config, tokenizer=tokenizer).cuda()
  model.train()

  # Match training: optimize frozen backbone + noise (EBM lives under backbone.ebm).
  opt = torch.optim.AdamW(
      itertools.chain(model.backbone.parameters(), model.noise.parameters()),
      lr=args.lr,
      betas=(config.optim.beta1, config.optim.beta2),
      eps=config.optim.eps,
      weight_decay=config.optim.weight_decay,
  )

  sentences = [
      "The quick brown fox jumps over the lazy dog.",
      "Manual NCE gradient check: pooling and readout should receive non-zero grads.",
  ]
  enc = tokenizer(
      sentences,
      return_tensors="pt",
      padding="max_length",
      truncation=True,
      max_length=args.length,
  )
  input_ids = enc["input_ids"].cuda()
  attention_mask = enc["attention_mask"].float().cuda()

  for step in range(args.steps):
    opt.zero_grad(set_to_none=True)
    losses = model._loss(input_ids, attention_mask, prefix="train")
    loss = losses.loss
    loss.backward()
    opt.step()

    ebm = model.ebm
    tm0 = ebm.token_mlp[0].weight.grad
    ts = ebm.token_score.weight.grad
    tg = ebm.token_gate.weight.grad
    vp = ebm.vocab_proj.weight.grad
    grad_bits = [
        f"token_mlp[0].weight: {tm0.norm().item():.6g}" if tm0 is not None else "token_mlp[0]: None",
        f"token_score.weight: {ts.norm().item():.6g}" if ts is not None else "token_score: None",
        f"token_gate.weight: {tg.norm().item():.6g}" if tg is not None else "token_gate: None",
        f"vocab_proj.weight: {vp.norm().item():.6g}" if vp is not None else "vocab_proj: None",
    ]

    print(f"step {step + 1}/{args.steps}  loss={loss.item():.6f}")
    print("  " + " | ".join(grad_bits))
    print(f"  ||grad|| (all ebm params): {_grad_norm(model.ebm.parameters()):.6f}")

  print("done.")


if __name__ == "__main__":
  main()
