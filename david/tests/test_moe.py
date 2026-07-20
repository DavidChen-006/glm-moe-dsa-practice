"""Row 3 test — the MoE path (TopkRouter + experts + shared expert). Arrange -> Act -> Assert.

Born RED (MoE forward produced nan — missing post_init/_init_weights), debugged
to green; now the regression guard for the MoE path.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architecture"))   # add david/architecture to the path

import torch
from transformers import GlmMoeDsaConfig

from modeling_glm_moe_dsa import GlmMoeDsaForCausalLM, GlmMoeDsaMoE

TOY = dict(
    vocab_size=6400,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
    moe_intermediate_size=64,
    n_routed_experts=8,
    num_experts_per_tok=2,
    mlp_layer_types=["dense", "sparse"],   # default is all-dense at 2 layers — MoE must be forced ON
)


def make_config():
    config = GlmMoeDsaConfig(**TOY)
    config._attn_implementation = "eager"
    return config


def make_moe(config):
    """Bare GlmMoeDsaMoE with params filled in. A bare module skips _init_weights
    (that only runs via the full model's post_init), so its torch.empty params are
    garbage — initialize here to test the LOGIC in isolation."""
    torch.manual_seed(0)
    moe = GlmMoeDsaMoE(config)
    for param in moe.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
    return moe


def test_router():
    # ---------- ARRANGE ----------
    # Contract per the reference (practice/glm copy): the router's forward returns
    # ONLY logits; top-k selection lives in GlmMoeDsaMoE.route_tokens_to_experts.
    config = make_config()
    moe = make_moe(config)                      # moe.gate is the TopkRouter
    x = torch.randn(2, 3, config.hidden_size)   # (batch=2, seq=3, hidden=128) -> 6 tokens

    # ---------- ACT ----------
    router_logits = moe.gate(x)                                        # step 1: score
    topk_indices, topk_weights = moe.route_tokens_to_experts(router_logits)   # step 2: select

    # ---------- ASSERT ----------
    n_tokens = 2 * 3
    assert router_logits.shape == (n_tokens, config.n_routed_experts), "one score per (token, expert)"
    assert topk_indices.shape == (n_tokens, config.num_experts_per_tok), "one row per token, k experts each"
    assert topk_weights.shape == (n_tokens, config.num_experts_per_tok), "one weight per chosen expert"
    assert (topk_indices >= 0).all() and (topk_indices < config.n_routed_experts).all(), "indices must be valid expert ids"
    assert torch.isfinite(topk_weights).all(), "routing weights must be finite"
    assert (topk_weights >= 0).all(), "sigmoid scores gathered as weights are never negative"


def test_moe_forward():
    # ---------- ARRANGE ----------
    config = make_config()
    moe = make_moe(config)
    x = torch.randn(2, 3, config.hidden_size)

    # ---------- ACT ----------
    out = moe(x)

    # ---------- ASSERT ----------
    assert out.shape == x.shape, "MoE must preserve shape (B, S, H) -> (B, S, H) — same contract as dense MLP"
    assert torch.isfinite(out).all(), "output must be all finite (no NaN/Inf)"


def test_moe_weights_initialized():
    # ---------- ARRANGE ----------
    # The reference builds expert weights with torch.empty (uninitialized memory)
    # and relies on _init_weights to fill them. If _init_weights misses them,
    # they are garbage — this test catches that at the source.
    torch.manual_seed(0)
    config = make_config()

    # ---------- ACT ----------
    model = GlmMoeDsaForCausalLM(config)   # full model, so _init_weights runs
    moe = model.model.layers[1].mlp
    assert isinstance(moe, GlmMoeDsaMoE), "layer 1 must be the sparse (MoE) layer"

    # ---------- ASSERT ----------
    for name, param in moe.named_parameters():
        assert torch.isfinite(param).all(), f"MoE parameter '{name}' contains nan/inf straight after init"


def test_moe_in_model_trains():
    # ---------- ARRANGE ----------
    torch.manual_seed(0)
    config = make_config()
    model = GlmMoeDsaForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))

    # ---------- ACT ----------
    out = model(input_ids, labels=input_ids)
    out.loss.backward()

    # ---------- ASSERT ----------
    assert torch.isfinite(out.loss), "loss through a sparse layer must be finite"
    router_grad = model.model.layers[1].mlp.gate.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0, "router must receive gradient (it must be learnable end-to-end)"


if __name__ == "__main__":
    test_router()
    test_moe_forward()
    test_moe_weights_initialized()
    test_moe_in_model_trains()
    print("PASS")
