"""demo_qkv.py — the THREE lines that make Q, K, V from the input tokens.

Each line: multiply the input x by a learned weight matrix -> a transformed view.
"""
import torch

# x = the input tokens (3 tokens, 4 features each)
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
])  # 3 x 4

# Three learned weight matrices (4x4 each) — in a real model these are TRAINED.
Wq = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
Wk = torch.tensor([[0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.]])
Wv = torch.tensor([[2., 0., 0., 0.], [0., 2., 0., 0.], [0., 0., 2., 0.], [0., 0., 0., 2.]])

# ===== THE THREE LINES =====
query_states = x @ Wq        # "what each token is looking for"
key_states   = x @ Wk        # "what each token offers"
value_states = x @ Wv        # "what each token gives if picked"

print("x (input tokens, 3x4):\n", x)
print("\nquery_states = x @ Wq:\n", query_states)
print("\nkey_states   = x @ Wk:\n", key_states)
print("\nvalue_states = x @ Wv:\n", value_states)
print("\nEach is the SAME 3 tokens, transformed 3 different ways by 3 trained matrices.")
