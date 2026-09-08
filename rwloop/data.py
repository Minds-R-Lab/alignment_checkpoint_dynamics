"""Data and model loading.

- `load_model(name, revision)`: HF causal LM in fp32 (bf16 optional) with the tokenizer.
- `eval_batches(...)`: a fixed, cached set of token batches so every experiment scores the same text.
- `token_stream(...)`: deterministic training stream (seeded shuffle buffer) so intervention and
  control runs see IDENTICAL data in IDENTICAL order — required for matched comparisons.

Default corpus is the deduplicated, uncopyrighted Pile mirror (Pythia's training distribution).
Set `--dataset` to any HF text dataset with a `text` column (e.g. HuggingFaceFW/fineweb, name=sample-10BT).
"""
from __future__ import annotations
import os, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CACHE = os.environ.get("RWLOOP_CACHE", "data_cache")


def load_model(name: str, revision: str | None = None, dtype: str = "float32", device: str = "cuda"):
    td = dict(float32=torch.float32, bfloat16=torch.bfloat16)[dtype]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=td)
    return model.to(device), tok


def _iter_text(dataset: str, config: str | None, split: str, seed: int, buffer: int):
    from datasets import load_dataset
    try:
        import zstandard  # noqa: F401  (the Pile mirrors are zstd-compressed jsonl)
    except ImportError as e:
        raise ImportError("pip install zstandard  (needed to stream zstd-compressed datasets such as the Pile)") from e
    ds = load_dataset(dataset, config, split=split, streaming=True)
    return ds.shuffle(seed=seed, buffer_size=buffer)


def token_stream(tok, dataset="monology/pile-uncopyrighted", config=None, split="train",
                 seq_len=2048, batch_size=32, seed=0, buffer=10_000, skip_docs=0):
    """Yield (batch_size, seq_len) LongTensors from a seeded, packed document stream."""
    ds = _iter_text(dataset, config, split, seed, buffer)
    buf = []
    for i, ex in enumerate(ds):
        if i < skip_docs:
            continue
        ids = tok(ex["text"]).input_ids + [tok.eos_token_id]
        buf.extend(ids)
        while len(buf) >= batch_size * seq_len:
            chunk = torch.tensor(buf[: batch_size * seq_len], dtype=torch.long).view(batch_size, seq_len)
            buf = buf[batch_size * seq_len:]
            yield chunk


def eval_batches(tok, n_tokens=1_000_000, seq_len=2048, batch_size=16, dataset="monology/pile-uncopyrighted",
                 config=None, split="train", seed=12345, skip_docs=200_000):
    """Fixed evaluation set, cached to disk keyed by (tokenizer, dataset, seed, n_tokens).
    skip_docs puts it far from the start of the training stream (stream seeds differ anyway)."""
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5(f"{tok.name_or_path}|{dataset}|{config}|{split}|{seed}|{n_tokens}|{seq_len}".encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"eval_{key}.npy")
    if os.path.exists(path):
        arr = np.load(path)
    else:
        need = n_tokens // seq_len
        rows = []
        for chunk in token_stream(tok, dataset, config, split, seq_len, batch_size, seed, skip_docs=skip_docs):
            rows.append(chunk.numpy())
            if sum(r.shape[0] for r in rows) >= need:
                break
        arr = np.concatenate(rows)[:need]
        np.save(path, arr)
    return [torch.from_numpy(arr[i:i + batch_size]) for i in range(0, len(arr), batch_size)]


def pythia_checkpoints(kind="standard") -> list[str]:
    """Revision names published for every Pythia model."""
    early = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if kind == "early":
        return [f"step{s}" for s in early]
    if kind == "coarse":
        return [f"step{s}" for s in [0, 1, 16, 64, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 143000]]
    return [f"step{s}" for s in early] + [f"step{s}" for s in range(1000, 144000, 1000)]
