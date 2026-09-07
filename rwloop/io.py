"""Result persistence: every script writes a single .npz/.json pair under results/<exp>/."""
import os, json, pickle
import numpy as np

def outdir(exp: str, root="results") -> str:
    d = os.path.join(root, exp); os.makedirs(d, exist_ok=True); return d

def save(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_json(obj, path: str):
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return o.item()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)
