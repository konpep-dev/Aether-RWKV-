"""
Training for RWKV v7 "Goose" — optimized for T4 16GB.
Usage: python train.py [--resume checkpoints/checkpoint_best.pt]
"""
import json, time, os, sys, argparse, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizer import ByteBPETokenizer as Tokenizer
from model import RWKV

# ── Config ──
DATASET_PATH = "aether_dataset_mixed.jsonl"
MODEL_SAVE_PATH = "aether_model.pt"
TOKENIZER_SAVE_PATH = "aether_tokenizer.json"
CKPT_DIR = "checkpoints"
BATCH_SIZE = 36
ACCUM_STEPS = 4  # Effective batch = 36 * 4 = 144
MAX_LENGTH = 256  # Covers full Q&A answers, fits T4 with batch 36
EPOCHS = 3
LEARNING_RATE = 3e-4
WARMUP_STEPS = 100
HIDDEN_SIZE = 640
NUM_LAYERS = 14
HEAD_SIZE = 64
DROPOUT = 0.1
SAVE_INTERVAL = 600
DRIVE_SAVE_INTERVAL = 3600
TEST_INTERVAL = 1000
DRIVE_DIR = os.environ.get("DRIVE_DIR", "")

TEST_PROMPTS = [
    "User: Hello, how are you?\n\nAether:",
    "User: What is machine learning?\n\nAether:",
    "User: Tell me about black holes.\n\nAether:",
    "User: Γεια σου, τι κάνεις;\n\nAether:",
    "User: Τι είναι η τεχνητή νοημοσύνη;\n\nAether:",
]

os.makedirs(CKPT_DIR, exist_ok=True)

def is_qa_sample(text):
    return "User:" in text and "Aether:" in text

def sample_tokenizer_texts(path, n=2000, qa_ratio=0.6, seed=42):
    """Sample mixed raw + Q&A lines for BPE (not just file-head raw text)."""
    import random
    rng = random.Random(seed)
    raw, qa = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            (qa if is_qa_sample(text) else raw).append(text)
    n_qa = min(len(qa), int(n * qa_ratio))
    n_raw = min(len(raw), n - n_qa)
    texts = rng.sample(qa, n_qa) + rng.sample(raw, n_raw)
    rng.shuffle(texts)
    return texts

def truncate_ids(ids, max_length, text, tokenizer):
    """For Q&A, prefer keeping the Aether: answer when truncating."""
    if len(ids) <= max_length:
        return ids
    if "Aether:" not in text:
        return ids[:max_length]
    pos = text.rfind("Aether:")
    prefix_ids = tokenizer.encode(text[:pos], add_bos=True, add_eos=False)
    suffix_ids = tokenizer.encode(text[pos:], add_bos=False, add_eos=True)
    if len(suffix_ids) >= max_length:
        return suffix_ids[:max_length]
    keep_prefix = max_length - len(suffix_ids)
    return prefix_ids[-keep_prefix:] + suffix_ids

def mask_before_assistant(tgt, ids, tokenizer):
    """Loss only on Aether: response tokens (ignore User: prefix)."""
    marker = tokenizer.encode("Aether:", add_bos=False, add_eos=False)
    if not marker:
        return tgt
    for i in range(len(ids) - len(marker) + 1):
        if ids[i:i + len(marker)] == marker:
            end = i + len(marker) - 1
            if end > 0:
                tgt[:end] = 0
            break
    return tgt

def build_interleaved_order(raw_list, qa_list, raw_per=2, qa_per=3):
    """All samples in order: 2 raw, 3 Q&A, repeat — no random shuffle."""
    order, ri, qi = [], 0, 0
    while ri < len(raw_list) or qi < len(qa_list):
        for _ in range(raw_per):
            if ri < len(raw_list):
                order.append(("raw", ri))
                ri += 1
        for _ in range(qa_per):
            if qi < len(qa_list):
                order.append(("qa", qi))
                qi += 1
    return order

# ── Dataset (ordered interleave, 100% of data) ──
class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.raw_samples, self.qa_samples = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                (self.qa_samples if is_qa_sample(text) else self.raw_samples).append(text)
        self.order = build_interleaved_order(self.raw_samples, self.qa_samples)
        total = len(self.raw_samples) + len(self.qa_samples)
        qa = len(self.qa_samples)
        print(f"Loaded ALL {total:,} samples ({self._estimate_mb():.1f}MB)")
        print(f"  Q&A: {qa:,} ({100*qa/total:.1f}%) | Raw: {len(self.raw_samples):,}")
        print(f"  Order: 2 raw -> 3 Q&A interleaved (shuffle=False)")

    def _estimate_mb(self):
        n = len(self.raw_samples) + len(self.qa_samples)
        sample = (self.raw_samples[:500] + self.qa_samples[:500])[:1000]
        return sum(len(s) for s in sample) * n / max(len(sample), 1) / 1024 / 1024

    def _get_text(self, idx):
        kind, i = self.order[idx]
        return self.qa_samples[i] if kind == "qa" else self.raw_samples[i]

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        text = self._get_text(idx)
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        ids = truncate_ids(ids, self.max_length, text, self.tokenizer)
        inp = torch.tensor(ids[:-1], dtype=torch.long)
        tgt = torch.tensor(ids[1:], dtype=torch.long)
        if is_qa_sample(text):
            tgt = mask_before_assistant(tgt, ids, self.tokenizer)
        return inp, tgt

def collate(batch):
    """Fast collation using pad_sequence"""
    inp, tgt = zip(*batch)
    # Use torch's built-in padding (much faster)
    from torch.nn.utils.rnn import pad_sequence
    pad_inp = pad_sequence(list(inp), batch_first=True, padding_value=0)
    pad_tgt = pad_sequence(list(tgt), batch_first=True, padding_value=0)
    return pad_inp, pad_tgt

# ── Periodic inference test ──
@torch.inference_mode()
def run_test(model, tokenizer, global_step):
    model.eval()
    device = next(model.parameters()).device
    print(f"\n{'='*50}")
    print(f"Test at step {global_step}:")
    for prompt in TEST_PROMPTS:
        ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        gen = model.generate(ids, max_new=40, temperature=0.5, top_k=20, repetition_penalty=1.1)
        new_ids = gen[len(ids):]
        out = tokenizer.decode(new_ids, skip_special=True)
        cut = out.find("User:")
        if cut != -1:
            out = out[:cut].rstrip()
        ascii_out = out.encode('ascii', errors='replace').decode('ascii')
        q_short = prompt.split("\n\nAether:")[0].replace("User: ", "")
        print(f"  Q: {q_short}")
        print(f"  A: {ascii_out}")
    print(f"{'='*50}\n")
    model.train()

# ── Copy checkpoint to Drive checkpoints/ folder ──
def save_to_drive(src_path, drive_dir=""):
    if not src_path or not os.path.exists(src_path) or not drive_dir:
        return
    try:
        dst_dir = os.path.join(drive_dir, "checkpoints")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_path, os.path.join(dst_dir, os.path.basename(src_path)))
        print(f"  [DRIVE] Copied {os.path.basename(src_path)}")
    except Exception as e:
        print(f"  [DRIVE] Copy failed: {e}")

# ── Training ──
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cpu':
        print("WARNING: No GPU! Training on CPU will be extremely slow (~100x slower than T4).")
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        print("cuDNN benchmark enabled, TF32 matmul precision=high")

    # Tokenizer — try loading pre-saved, else train with 2000 texts
    tokenizer = Tokenizer(vocab_size=8192)
    if os.path.exists(TOKENIZER_SAVE_PATH):
        tokenizer.load(TOKENIZER_SAVE_PATH)
        print(f"Loaded tokenizer — vocab={tokenizer.get_vocab_size()}")
    else:
        texts = sample_tokenizer_texts(DATASET_PATH, n=2000, qa_ratio=0.6)
        tokenizer.train(texts, max_texts=2000)
        tokenizer.save(TOKENIZER_SAVE_PATH)
        print(f"Trained tokenizer — vocab={tokenizer.get_vocab_size()}")
        del texts
    vocab_size = tokenizer.get_vocab_size()

    # Dataset
    dataset = TextDataset(DATASET_PATH, tokenizer, MAX_LENGTH)
    num_workers = 0 if os.name == 'nt' else 4  # More workers for data loading
    prefetch_factor = 2 if num_workers > 0 else None
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate, num_workers=num_workers,
                        pin_memory=(device=='cuda'), drop_last=True,
                        prefetch_factor=prefetch_factor, persistent_workers=(num_workers > 0))
    print(f"Batches/epoch: {len(loader)} (num_workers={num_workers})")

    # Model
    model = RWKV(vocab_size=vocab_size, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, head_size=HEAD_SIZE, dropout=DROPOUT)
    model.to(device)
    
    # Enable torch.compile for faster training (if available)
    if device == 'cuda':
        compiled = model.compile_model(mode="default")
        if compiled:
            print("torch.compile enabled for WKV kernel")

    # Optimizer
    decay = [p for n,p in model.named_parameters() if 'weight' in n and p.ndim >= 2]
    no_decay = [p for n,p in model.named_parameters() if not ('weight' in n and p.ndim >= 2)]
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': 0.1},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    # Resume
    start_epoch = 0
    start_batch = 0
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        start_batch = ckpt.get('batch', 0)
        best_loss = ckpt.get('loss', float('inf'))
        print(f"Resumed epoch {start_epoch+1}, batch {start_batch}, loss {best_loss:.4f}")

    total_steps = len(loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5)

    print(f"\nTraining: {BATCH_SIZE} micro-batch, {ACCUM_STEPS} accum -> {BATCH_SIZE*ACCUM_STEPS} effective")
    print(f"Total steps: {total_steps}, Warmup: {WARMUP_STEPS}, Test every {TEST_INTERVAL} steps")

    model.train()
    global_step = 0
    last_save = time.time()
    last_drive_save = time.time()
    t_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (inp, tgt) in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(inp)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                    loss = loss / ACCUM_STEPS
                scaler.scale(loss).backward()
            else:
                logits, _ = model(inp)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
                loss = loss / ACCUM_STEPS
                loss.backward()

            epoch_loss += loss.item() * ACCUM_STEPS
            num_batches += 1

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                global_step += 1
                
                if global_step < WARMUP_STEPS:
                    scale = min(1.0, global_step / WARMUP_STEPS)
                    for pg in optimizer.param_groups:
                        pg['lr'] = LEARNING_RATE * scale

                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if (batch_idx + 1) % 20 == 0:
                avg_loss = epoch_loss / max(num_batches, 1)
                elapsed = time.time() - t_start
                actual_steps = global_step
                sps = actual_steps / max(elapsed, 1)
                remain = (total_steps // ACCUM_STEPS - actual_steps) / max(sps, 0.01)
                print(f"E{epoch+1}/{EPOCHS} B{batch_idx+1}/{len(loader)} | "
                      f"loss: {avg_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e} | "
                      f"{sps:.1f} step/s | ETA: {remain/3600:.1f}h")

            # Periodic test
            if global_step > 0 and global_step % TEST_INTERVAL == 0:
                run_test(model, tokenizer, global_step)

            # Periodic save
            if time.time() - last_save >= SAVE_INTERVAL:
                avg_loss = epoch_loss / max(num_batches, 1)
                ckpt_path = os.path.join(CKPT_DIR,
                    f"checkpoint_e{epoch+1}_b{batch_idx+1}_loss{avg_loss:.3f}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch, 'batch': batch_idx + 1,
                    'loss': avg_loss, 'vocab_size': vocab_size,
                    'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS,
                    'head_size': HEAD_SIZE,
                }, ckpt_path)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    ckpt_best = os.path.join(CKPT_DIR, "checkpoint_best.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocab_size': vocab_size, 'hidden_size': HIDDEN_SIZE,
                        'num_layers': NUM_LAYERS, 'head_size': HEAD_SIZE,
                        'loss': avg_loss, 'epoch': epoch, 'batch': batch_idx + 1,
                    }, ckpt_best)
                    save_to_drive(ckpt_best, DRIVE_DIR)
                last_save = time.time()
                print(f"  [SAVED] loss={avg_loss:.4f}")

            # Periodic Drive upload (every hour regardless of local save)
            if time.time() - last_drive_save >= DRIVE_SAVE_INTERVAL:
                ckpt_hourly = os.path.join(CKPT_DIR,
                    f"checkpoint_hourly_e{epoch+1}_b{batch_idx+1}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab_size': vocab_size, 'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS, 'head_size': HEAD_SIZE,
                    'loss': epoch_loss / max(num_batches, 1),
                    'epoch': epoch, 'batch': batch_idx + 1,
                }, ckpt_hourly)
                save_to_drive(ckpt_hourly, DRIVE_DIR)
                last_drive_save = time.time()

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size, 'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS, 'head_size': HEAD_SIZE,
        'dropout': DROPOUT,
    }, MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    elapsed = (time.time() - t_start) / 3600
    print(f"Total time: {elapsed:.1f}h")

    # Final test
    run_test(model, tokenizer, total_steps)

if __name__ == "__main__":
    train()
