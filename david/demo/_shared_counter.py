"""A MODULE — one shared namespace for the whole program. Imported by demo_module_vs_class.py.
Its `count` is module-level state: there is exactly ONE of it, program-wide."""
count = 0


def bump():
    global count
    count += 1


def value():
    return count
