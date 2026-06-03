# Aether

> A 25M-parameter RWKV language model that runs on CPU.

A small, self-contained language model using the **RWKV v4** architecture — a linear-time attention design with no KV cache and no GPU requirement. Trained on a custom dataset of 22,000 computer science and casual conversations (English + Greek).

The trained weights are not included in this repository. See [Training your own](#training-your-own).

---

## Highlights

- **24.7M parameters**, 12 layers, 480 hidden, 1920 feed-forward
- **O(T) sequence length** — runs as fast as a plain RNN at inference
- **Word-level tokenizer**, 8K vocab
- **22K conversations** of mixed English + Greek training data
- Single-file PyTorch implementation, no external dependencies beyond torch + psutil

---

## Architecture

Each of the 12 blocks contains:

- **Time-decay channel** — per-channel learned forgetting rate
- **Token-shift** — AFT-style blend with the previous token's hidden state
- **WKV attention** — the O(T) "softmax-like" recurrence over a numerator/denominator state
- **Receptance gate** — sigmoid yes/no on whether the new information flows through

The full per-equation write-up lives on the [project page](https://USERNAME.github.io/aether/).

```
x'    = x · tm + x_prev · (1 - tm)              # token-shift
wkv_t = (e^(u+k_t)·v_t + num_{t-1}) / (e^(u+k_t) + den_{t-1})   # WKV
num_t = w · num_{t-1} + e^k_t · v_t             # numerator state
den_t = w · den_{t-1} + e^k_t                   # denominator state
y_t   = σ(r_t) · wkv_t                          # output gate
```

---

## Repository

| File | Purpose |
| --- | --- |
| `index.html` | GitHub Pages site (architecture deep-dive + dataset preview) |
| `aether_dataset.jsonl` | Training data, 22K conversations, ~2.7 MB |
| `aether_tokenizer.json` | Word-level tokenizer, 8K vocab |
| `generate_dataset.py` | Programmatic dataset generator |
| `tokenizer.py` | Tokenizer implementation |
| `model.py` | RWKV v4 model definition |
| `train.py` | Training loop with resume + SystemMonitor |
| `inference.py` | Generation with live ASCII network visualizer |

---

## Training your own

```bash
# 1. Generate the dataset (only if you want to regenerate it)
python generate_dataset.py

# 2. Tokenize + train (~15 hours on a 4-core CPU)
python train.py

# 3. Run inference
python inference.py --quick "Hello, Aether."
```

Training is single-threaded by design. On an Intel i5-4570, expect ~12 samples/sec and roughly 12-14 epochs in a 15-hour session. Checkpoints are auto-saved as `checkpoint_best.pt` and `checkpoint_interrupted.pt`; the training script supports resume from either.

---

## Inference

```python
from inference import load_model, generate

model, tokenizer, device = load_model("aether_model.pt")
out = generate(model, tokenizer, "User: What is RWKV?\n\nAether:", max_tokens=120)
print(out)
```

A live ASCII visualization of the network is included in `inference.py` — it animates the activations of all 12 blocks in your terminal as the model generates, throttled to the display refresh rate.

---

## License

MIT. See [LICENSE](LICENSE).
