"""Contract check for token-additive readout (no diffusion imports)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, H = 2, 16, 32
x = torch.randn(B, T, H)
token_mlp = nn.Sequential(
  nn.Linear(H, H, bias=True),
  nn.ReLU(),
  nn.Linear(H, H, bias=True),
)
token_score = nn.Linear(H, 1, bias=True)
token_gate = nn.Linear(H, 1, bias=True)
h = token_mlp(x)
token_contrib = (torch.sigmoid(token_gate(h)) * F.softplus(token_score(h))).squeeze(-1)
energy = token_contrib.sum(dim=-1, keepdim=True)
assert energy.shape == (B, 1)
assert token_contrib.shape == (B, T)
assert (token_contrib >= 0).all()
print("smoke_ebm_token_additive_readout ok")
