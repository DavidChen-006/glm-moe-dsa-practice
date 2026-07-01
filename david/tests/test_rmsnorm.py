"""Row 1 test — GlmMoeDsaRMSNorm alone.

Pattern to copy for rows 2-6: build the class, feed a tensor, print input->output shape
(and here, per-row RMS, which should all become ~1). Structured as Arrange -> Act -> Assert.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))   # add david/ to the path

import torch
from modeling_glm_moe_dsa import GlmMoeDsaRMSNorm


def per_row_rms(t):   # helper: the "size" of each token row
    return [round(r, 3) for r in t.pow(2).mean(-1).sqrt().flatten().tolist()]


# ---------- ARRANGE ---------- (build the thing under test + its input)
H = 8
norm = GlmMoeDsaRMSNorm(H)
x = torch.randn(2, 3, H) * 5           # (batch=2, seq=3, hidden=8), deliberately varied scale

# ---------- ACT ---------- (run the one operation being tested)
out = norm(x)

# ---------- ASSERT ---------- (check the properties that must hold)
print("input  shape:", tuple(x.shape))
print("output shape:", tuple(out.shape), " (should MATCH input)")
print("per-row RMS before:", per_row_rms(x))
print("per-row RMS after :", per_row_rms(out), " (should all be ~1.0)")

assert out.shape == x.shape, "shape must be preserved"
assert all(abs(r - 1.0) < 1e-3 for r in per_row_rms(out)), "each row should be RMS 1"
print("\nPASS: RMSNorm preserves shape and normalizes each row to RMS 1.")
