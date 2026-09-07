#!/usr/bin/env python
"""Exp 6 — toy transformer (CPU-capable) reproducing: Phase-1 sign prediction from q = a^T M a,
component intervention (c->f) and bias intervention (f->c) with matched data order. Multi-seed.

  python scripts/06_toy.py --style pythia --steps 1500 --intervene_at 700 --arm control --seed 0
  python scripts/06_toy.py --style pythia --arm component:2 --seed 0
  python scripts/06_toy.py --style smol   --arm bias:1.0   --seed 0
styles: pythia (parallel residual, GeLU) | smol (sequential, gated SiLU)
Downloads tiny-shakespeare on first use (set --text to any UTF-8 file for other corpora).
Output: results/06_toy/<style>_<arm>_seed<k>.pkl, analyzable with 05_analyze_interventions.py --indir results/06_toy --model toy --revision <style>
"""
import argparse, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from rwloop.io import outdir, save

p = argparse.ArgumentParser()
p.add_argument("--style", default="pythia", choices=["pythia", "smol"]); p.add_argument("--arm", default="control")
p.add_argument("--steps", type=int, default=1500); p.add_argument("--intervene_at", type=int, default=700)
p.add_argument("--seed", type=int, default=0); p.add_argument("--d", type=int, default=128); p.add_argument("--mlp", type=int, default=512)
p.add_argument("--layers", type=int, default=3); p.add_argument("--T", type=int, default=128); p.add_argument("--B", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--window", type=int, default=30)
p.add_argument("--snap_after", nargs="+", type=int, default=[0, 30, 200, 450, 800])
p.add_argument("--text", default="data_cache/tinyshakespeare.txt"); p.add_argument("--threads", type=int, default=4)
args = p.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(args.threads)
os.makedirs(os.path.dirname(args.text), exist_ok=True)
if not os.path.exists(args.text):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", args.text)
txt = open(args.text).read(); chars = sorted(set(txt)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in txt]); V = len(chars)
D, NH, NL, MLP, T, B = args.d, 4, args.layers, args.mlp, args.T, args.B


class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1 = nn.LayerNorm(D); s.ln2 = nn.LayerNorm(D); s.attn = nn.MultiheadAttention(D, NH, batch_first=True)
        s.up = nn.Linear(D, MLP); s.down = nn.Linear(MLP, D)
        s.gate = nn.Linear(D, MLP, bias=False) if args.style == "smol" else None
        s.vbias = nn.Parameter(torch.zeros(MLP), requires_grad=False)      # virtual bias for bias arms
        for m in s.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
        s.x_in = None; s.g_out = None; s._pre = None
    def mlp(s, x):
        s.x_in = x.detach()
        if args.style == "pythia":
            pre = s.up(x) + s.vbias; s._pre = pre.detach(); h = F.gelu(pre)
        else:
            gpre = s.gate(x) + s.vbias; s._pre = gpre.detach(); h = F.silu(gpre) * s.up(x)
        out = s.down(h)
        if out.requires_grad: out.register_hook(lambda g: setattr(s, "g_out", g.detach()))
        return out
    def forward(s, x, mask):
        a = s.ln1(x)
        if args.style == "pythia":
            return x + s.attn(a, a, a, attn_mask=mask, need_weights=False)[0] + s.mlp(s.ln2(x))
        x = x + s.attn(a, a, a, attn_mask=mask, need_weights=False)[0]; return x + s.mlp(s.ln2(x))


class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, D); s.pos = nn.Embedding(T, D); s.blocks = nn.ModuleList([Block() for _ in range(NL)])
        s.lnf = nn.LayerNorm(D); s.head = nn.Linear(D, V, bias=False)
        for e in (s.emb, s.pos, s.head): nn.init.normal_(e.weight, std=0.02)
    def forward(s, idx):
        x = s.emb(idx) + s.pos(torch.arange(idx.shape[1])); mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        for b in s.blocks: x = b(x, mask)
        return s.head(s.lnf(x))


m = LM(); opt = torch.optim.AdamW([p for n, p in m.named_parameters() if "vbias" not in n], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
def batch():
    ix = torch.randint(len(data) - T - 1, (B,)); return torch.stack([data[i:i + T] for i in ix]), torch.stack([data[i + 1:i + T + 1] for i in ix])
def cosines(): return [F.cosine_similarity(b.up.weight.detach(), b.down.weight.detach().T, dim=1).numpy() for b in m.blocks]
Macc = [None] * NL
def accum_M():
    for i, b in enumerate(m.blocks):
        x = b.x_in.reshape(-1, D); g = b.g_out.reshape(-1, D); Mi = (g.T @ x) / len(x)
        Macc[i] = Mi if Macc[i] is None else 0.95 * Macc[i] + 0.05 * Mi
def q_values():
    out = []
    for i, b in enumerate(m.blocks):
        Mi = Macc[i]; A = b.up.weight.detach(); q = ((A @ Mi) * A).sum(1) / (A * A).sum(1)
        out.append((q.mean() / Mi.norm() * D).item())
    return out
snap_steps = sorted({args.intervene_at + d for d in args.snap_after} | {args.steps}); total = max(snap_steps)
snaps, FREQ, EXPO, state = {}, {}, {}, {"groups": None, "cos_pre": None}
t0 = time.time()
for step in range(1, total + 1):
    xb, yb = batch(); logits = m(xb); loss = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1)); opt.zero_grad(); loss.backward(); accum_M()
    if any(s - args.window < step <= s for s in snap_steps):
        FREQ["n"] = FREQ.get("n", 0) + 1
        for i, b in enumerate(m.blocks):
            FREQ[i] = FREQ.get(i, 0) + (b._pre > 0).float().mean((0, 1))
            h = (F.gelu(b._pre) if args.style == "pythia" else F.silu(b._pre)).reshape(-1, MLP); g = b.g_out.reshape(-1, D)
            A = b.up.weight.detach(); Ah = A / A.norm(dim=1, keepdim=True); EXPO[i] = EXPO.get(i, 0) + ((h.T @ g) * Ah).sum(1) / len(g)
    opt.step()
    if step in snap_steps:
        snaps[step] = dict(mlp=cosines(), firing=[(FREQ[i] / FREQ["n"]).numpy() for i in range(NL)],
                           exposure=[(EXPO[i] / FREQ["n"]).numpy() for i in range(NL)], q=q_values(), loss=loss.item())
        FREQ, EXPO = {}, {}
    if step == args.intervene_at:
        state["cos_pre"] = dict(mlp=cosines()); kind = args.arm.split(":")
        with torch.no_grad():
            if kind[0] == "component":
                for b in m.blocks[1:]:
                    A = b.up.weight; Ah = A / A.norm(dim=1, keepdim=True); Dw = b.down.weight.T; proj = (Dw * Ah).sum(1, keepdim=True) * Ah
                    b.down.weight.copy_((Dw + float(kind[1]) * proj).T)
            elif kind[0] == "bias":
                beta = float(kind[1]); g = torch.Generator().manual_seed(args.seed * 1000 + 123); state["groups"] = {}
                for li, b in enumerate(m.blocks):
                    if li == 0: continue
                    perm = torch.randperm(MLP, generator=g); up_idx, dn_idx = perm[:MLP // 5], perm[MLP // 5: 2 * MLP // 5]
                    sd = b._pre.std().item(); b.vbias.requires_grad_(True); b.vbias[up_idx] += beta * sd; b.vbias[dn_idx] -= beta * sd
                    grp = np.zeros(MLP, int); grp[up_idx.numpy()] = 1; grp[dn_idx.numpy()] = -1; state["groups"][li] = grp
                opt.add_param_group({"params": [b.vbias for b in m.blocks[1:]], "weight_decay": 0.0})
        print(f"intervened ({args.arm}) at {step}: cos {[round(float(c.mean()), 3) for c in cosines()]}", flush=True)
    if step % 100 == 0 or step == total:
        print(f"step {step:5d} loss {loss.item():.3f} | " + " | ".join(f"L{i}: q {q:+.2f} cos {c.mean():+.3f}" for i, (q, c) in enumerate(zip(q_values(), cosines()))) + f" [{time.time()-t0:.0f}s]", flush=True)
out = dict(model="toy", revision=args.style, arm=args.arm, seed=args.seed, intervene_at=args.intervene_at, layers=list(range(1, NL)),
           snaps=snaps, groups=state["groups"], cos_pre=state["cos_pre"], losses=[], eval_loss=loss.item())
save(out, os.path.join(outdir("06_toy"), f"toy_{args.style}_{args.arm.replace(':', '_')}_seed{args.seed}.pkl"))
