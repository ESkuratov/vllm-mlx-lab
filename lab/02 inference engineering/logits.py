from mlx_lm import load
import mlx.core as mx

model, tok = load ("mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")
ids = mx.array(tok.encode("2+2 equals"))
logits = model(ids[None])
last = logits[0, -1]
probs = mx.softmax(last)
for t in mx.argsort(-last)[:5].tolist():
    print(f"{tok.decode([t])!r:12} logit={float(last[t]):8.3f}  =p{float(probs[t]):.4f}")
