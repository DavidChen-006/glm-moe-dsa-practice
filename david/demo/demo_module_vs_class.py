"""demo_module_vs_class.py — the difference, shown by identity and state.

MODULE = a singleton: imported once, ONE shared copy for the whole program.
CLASS  = a cookie-cutter: instantiate it many times, each a SEPARATE object.
"""
import importlib
import sys

import _shared_counter as a       # a MODULE (the .py file next door)


class Counter:                    # a CLASS (a blueprint)
    def __init__(self):
        self.count = 0            # per-INSTANCE state
    def bump(self):
        self.count += 1
    def value(self):
        return self.count


# ================================================================ IDENTITY
print("=" * 66)
print("IDENTITY — how many objects do you actually get?")
print("=" * 66)
b = importlib.import_module("_shared_counter")   # grab the module AGAIN at runtime...
print("import module twice -> a is b ?  ", a is b)
print("   (Python cached it in sys.modules; you get the SAME object back)")
print("   sys.modules has it?", "_shared_counter" in sys.modules)

c1 = Counter()
c2 = Counter()
print("\ninstantiate class twice -> c1 is c2 ?  ", c1 is c2)
print("   (each Counter() runs __init__ fresh -> a NEW object every time)")

# ================================================================ STATE
print("\n" + "=" * 66)
print("STATE — change one 'copy'; does the other see it?")
print("=" * 66)

print("\nMODULE (shared): bump through name 'a' twice, then read via 'b':")
a.bump()
a.bump()
print(f"   a.value() = {a.value()}    b.value() = {b.value()}   <- BOTH 2. one shared state.")

print("\nCLASS (independent): bump c1 twice, then read c2:")
c1.bump()
c1.bump()
print(f"   c1.value() = {c1.value()}    c2.value() = {c2.value()}   <- 2 and 0. separate states.")

# ================================================================ VERDICT
print("\n" + "=" * 66)
print("the one-line difference")
print("=" * 66)
print("""   MODULE = ONE thing everyone shares (a singleton namespace).
            no instances. mutate it -> the whole program sees it.
   CLASS  = a BLUEPRINT for many independent things.
            each instance owns its own state; mutate one -> others untouched.

   in your code:
     torch.autograd  -> MODULE: one engine, every tensor calls the same one
     PretrainDataset -> CLASS:  train_ds and val_ds are two independent instances
     GlmMoeDsaConfig -> CLASS:  each model gets its own config object
""")
