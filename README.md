# GLM-5.2 From Scratch

Rebuilding the **GLM-5.2** architecture from scratch, at toy scale, to understand it down to the bolts — the model, the training loop, and the calculus underneath backpropagation.

The method is **reverse learning**: start from the real released architecture, stand it up as a black box, then peel one component at a time until you hit the raw math — reimplementing each piece yourself and proving it works. Same code as real GLM-5.2; only the config numbers and the data are toy-sized.

> Toy `hidden_size=128`, `num_layers=2`, `vocab=6400` on a MacBook (`mps`) — vs. real GLM-5.2's `hidden_size=6144`, `78` layers, `256` experts, `154,880` vocab. Identical architecture, different scale. The only thing between the two is compute.

## What's in here

| Path | What it is |
|------|-----------|
| **`david/`** | **All my own work** — the from-scratch rebuild |
| `glm/` | Reference: the real `transformers` GLM-MoE-DSA implementation (the target) |
| `minimind/` | Reference: [jingyaogong/minimind](https://github.com/jingyaogong/minimind) — a readable modern training loop |

### Inside `david/`

| Path | Contents |
|------|----------|
| `architecture/modeling_glm_moe_dsa.py` | The model — RMSNorm, Attention (+RoPE), MLP, DecoderLayer, Model, ForCausalLM, all written by hand |
| `inference.py` | Text→text pipeline (mirrors HF's `TextGenerationPipeline`) with CLI + REPL modes |
| `training/` | `train.py` (the loop), `prepare_data.py`, `lm_dataset.py`, `prepare_jsonl.py` — pretraining on tiny-shakespeare |
| `tests/` | Bottom-up AAA pytest suite, one test per model class (run in CI on every push) |
| `demo/` | ~45 standalone demo scripts, each visualizing one concept (RoPE rotation, the causal mask, the backward graph, cross-entropy, gradient descent…) |
| `notes/` | Written notes tracing the mathematical heritage of backprop — Leibniz (the chain rule), Euler (functions vs. values) |
| `visualizations/` | Interactive HTML of the full backward computation graph |

## The ladder

Peel from the roof down to the foundation. Each rung: understand a component, reimplement it, swap it into the real model — outputs must match.

- ✅ **Spine** — `embed → [norm → attention → norm → MLP] × N → norm → lm_head`, the residual stream, end-to-end inference (ids → logits → predicted token)
- ✅ **RoPE** — rotary position embeddings on Q and K
- ✅ **Training** — cross-entropy, backprop, AdamW, cosine LR schedule; pretraining on char-level / BPE tiny-shakespeare
- ⬜ **MoE** — the mixture-of-experts router + experts (256 experts, top-8 in real GLM)
- ⬜ **MLA** — multi-head latent attention (compressed KV cache)
- ⬜ **DSA** — DeepSeek sparse attention (top-k token selection)
- ⬜ **Capstone** — load real released weights into the rebuilt model, confirm logits match

## Run it

```bash
# inference (untrained → gibberish; trained → Shakespeare-ish)
cd david && python inference.py --prompt "Before we proceed"

# prepare data + train the toy model
cd david/training && python prepare_data.py && python train.py

# the test suite
cd david && python -m pytest tests/ -q

# any demo, e.g. how RoPE rotates a vector
python david/demo/demo_rope.py
```

## Beyond the code

A running theme is refusing to treat any layer as magic. The `notes/` and `demo/` folders chase backpropagation all the way down — not just *how* `loss.backward()` works (the recorded graph, the per-weight shadow gradients, the chain-rule walk) but *why*, through the history of the ideas: Leibniz's chain rule, Euler's distinction between a function and its value, Cauchy's gradient descent.

---

*A learning project. The reference implementations under `glm/` and `minimind/` are not mine — see their upstreams for licensing.*
