"""
ΕΚΠΑΙΔΕΥΣΗ (TRAINING) — Μαθαίνουμε στο Aether να μιλάει

Διαδικασία:
1. Φορτώνουμε το dataset (aether_dataset.jsonl)
2. Χτίζουμε tokenizer (λέξεις → αριθμούς)
3. Δημιουργούμε το μοντέλο RWKV (~25M params)
4. Εκπαιδεύουμε με backpropagation
5. Αποθηκεύουμε το εκπαιδευμένο μοντέλο
"""

import json
import time
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import psutil
import ctypes

from tokenizer import Tokenizer
from model import RWKV


# ---------- WIN32 CONSOLE STRUCTS ----------

class COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

class SMALL_RECT(ctypes.Structure):
    _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short),
                ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]

class CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
    _fields_ = [("dwSize", COORD), ("dwCursorPosition", COORD),
                ("wAttributes", ctypes.c_ushort), ("srWindow", SMALL_RECT),
                ("dwMaximumWindowSize", COORD)]


# ---------- LIVE SYSTEM MONITOR ----------

class SystemMonitor:
    def __init__(self, height=8, bar_w=6):
        self.height = height
        self.bar_w = bar_w
        self.header = " CPU" + " " * (bar_w - 2) + "RAM"
        self._last_update = 0.0
        self._interval = 2.0
        self._wmi = None
        self._try_wmi()
        self._use_win32 = self._detect_win32()
        self._use_ansi = self._detect_ansi()
        self._old_x = -1
        self._old_y = -1
        self._old_w = 0
        self._old_h = 0

    def _try_wmi(self):
        try:
            import wmi
            self._wmi = wmi.WMI()
        except ImportError:
            pass

    def _detect_ansi(self):
        if not sys.stdout.isatty():
            return False
        if 'WT_SESSION' in os.environ or 'TERM_PROGRAM' in os.environ:
            return True
        try:
            h = ctypes.windll.kernel32.GetStdHandle(-11)
            m = ctypes.c_uint()
            return bool(ctypes.windll.kernel32.GetConsoleMode(h, ctypes.byref(m)))
        except Exception:
            return False

    def _detect_win32(self):
        try:
            h = ctypes.windll.kernel32.GetStdHandle(-11)
            return h is not None and h != -1
        except Exception:
            return False

    def _temp(self):
        if self._wmi is None:
            return "N/A"
        try:
            tz = self._wmi.MSAcpi_ThermalZoneTemperature()
            if tz:
                c = tz[0].CurrentTemperature / 10.0 - 273.15
                return f"{c:.0f}C"
        except Exception:
            pass
        return "N/A"

    def _data(self):
        cpu = psutil.cpu_percent(interval=0)
        ram = psutil.virtual_memory().percent
        return cpu, ram, self._temp()

    def _make(self, cpu, ram, t):
        lines = [self.header]
        for i in range(self.height):
            cf = int(cpu / 100 * self.height)
            rf = int(ram / 100 * self.height)
            cl = "#" if i >= (self.height - cf) else "."
            rl = "#" if i >= (self.height - rf) else "."
            lines.append(" " + cl * self.bar_w + " " + rl * self.bar_w)
        lines.append(f" {cpu:>3.0f}%    {ram:>3.0f}%   {t}")
        return lines

    def draw(self):
        c, r, t = self._data()
        for l in self._make(c, r, t):
            print(l)
        sys.stdout.flush()

    def update(self):
        now = time.time()
        if now - self._last_update < self._interval:
            return
        self._last_update = now
        c, r, t = self._data()
        lines = self._make(c, r, t)
        ok = False
        if self._use_ansi:
            ok = self._draw_ansi(lines)
        if not ok and self._use_win32:
            ok = self._draw_win32(lines)
        if not ok:
            for l in lines:
                print(l)
            sys.stdout.flush()

    def _get_console_info(self):
        h = ctypes.windll.kernel32.GetStdHandle(-11)
        info = CONSOLE_SCREEN_BUFFER_INFO()
        ctypes.windll.kernel32.GetConsoleScreenBufferInfo(h, ctypes.byref(info))
        return h, info

    def _draw_win32(self, lines):
        try:
            h, info = self._get_console_info()
            saved = info.dwCursorPosition
            bot = info.srWindow.Bottom
            right = info.srWindow.Right
            mw = max(len(l) for l in lines)
            mh = len(lines)
            x = max(0, right - mw + 1)
            y = max(0, bot - mh + 1)
            # Clear old area
            if self._old_x >= 0:
                for row in range(self._old_h):
                    coord = COORD(self._old_x, self._old_y + row)
                    ctypes.windll.kernel32.SetConsoleCursorPosition(h, coord)
                    written = ctypes.c_ulong()
                    ctypes.windll.kernel32.WriteConsoleA(
                        h, (" " * self._old_w).encode(), self._old_w,
                        ctypes.byref(written), None
                    )
            # Draw new
            written = ctypes.c_ulong()
            for i, line in enumerate(lines):
                coord = COORD(x, y + i)
                ctypes.windll.kernel32.SetConsoleCursorPosition(h, coord)
                ctypes.windll.kernel32.WriteConsoleA(
                    h, line.ljust(mw).encode(), mw,
                    ctypes.byref(written), None
                )
            ctypes.windll.kernel32.SetConsoleCursorPosition(h, saved)
            self._old_x, self._old_y = x, y
            self._old_w, self._old_h = mw, mh
            return True
        except Exception:
            return False

    def _draw_ansi(self, lines):
        try:
            tw, th = shutil.get_terminal_size()
            mw = max(len(l) for l in lines)
            mh = len(lines)
            x = max(1, tw - mw + 1)
            y = max(1, th - mh + 1)
            # Save cursor, position, draw, restore
            sys.stdout.write("\0337")
            for i, line in enumerate(lines):
                sys.stdout.write(f"\033[{y + i};{x}H" + line.ljust(mw) + "\033[K")
            sys.stdout.write("\0338")
            sys.stdout.flush()
            return True
        except Exception:
            return False


# ---------- 1. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ ----------

class TextDataset(Dataset):
    """
    Dataset: μετατρέπει τα δεδομένα μας σε μορφή που καταλαβαίνει το PyTorch
    
    Κάθε δείγμα είναι ένα κομμάτι κειμένου (π.χ. "User: Hello\n\nAether: Hi!")
    Το μοντέλο μαθαίνει να προβλέπει την ΕΠΟΜΕΝΗ λέξη από τις προηγούμενες
    """
    
    def __init__(self, jsonl_path, tokenizer, max_length=64):
        """
        Φορτώνει το dataset από αρχείο JSONL
        
        Παράμετροι:
            jsonl_path: διαδρομή στο aether_dataset.jsonl
            tokenizer: ο tokenizer για μετατροπή κειμένου σε IDs
            max_length: μέγιστο μήκος ακολουθίας (κόβουμε μεγαλύτερες)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Διαβάζουμε όλες τις συνομιλίες από το JSONL
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data["text"]
                self.samples.append(text)
        
        print(f"Loaded {len(self.samples)} conversations")
    
    def __len__(self):
        """Πόσα δείγματα έχει το dataset"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Παίρνει ένα δείγμα και το μετατρέπει σε:
        - input_ids: οι λέξεις σαν αριθμοί (για είσοδο)
        - target_ids: η επόμενη λέξη σε κάθε θέση (για έξοδο)
        
        Π.χ. αν το κείμενο είναι "Hello world"
        input_ids:  [BOS, Hello, world, EOS]
        target_ids: [Hello, world, EOS, PAD]
        """
        text = self.samples[idx]
        
        # Μετατροπή κειμένου σε IDs (με BOS/EOS)
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        
        # Κόβουμε αν είναι πολύ μεγάλο
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        
        # Δημιουργούμε input (όλες εκτός τελευταίας) και target (όλες εκτός πρώτης)
        input_ids = ids[:-1]  # Π.χ. [BOS, Hello, world]
        target_ids = ids[1:]  # Π.χ. [Hello, world, EOS]
        
        # Μετατροπή σε tensors (πίνακες PyTorch)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        
        return input_ids, target_ids


def collate_batch(batch):
    """
    Συνενώνει πολλά δείγματα σε ένα batch (ομάδα)
    
    Επειδή κάθε δείγμα μπορεί να έχει διαφορετικό μήκος,
    συμπληρώνουμε (padding) τα μικρότερα στο μήκος του μεγαλύτερου
    """
    input_ids_list, target_ids_list = zip(*batch)
    
    # Βρίσκουμε το μέγιστο μήκος
    max_len = max(len(ids) for ids in input_ids_list)
    
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(input_ids_list, target_ids_list):
        # Πόσο padding χρειάζεται
        pad_len = max_len - len(inp)
        
        # Συμπληρώνουμε με PAD tokens (ID = 0)
        padded_inp = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
        padded_tgt = torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)])
        
        padded_inputs.append(padded_inp)
        padded_targets.append(padded_tgt)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)


# ---------- 2. ΕΚΠΑΙΔΕΥΣΗ ----------

def train_model(model, dataloader, epochs=10, lr=0.001, device='cpu',
                checkpoint_dir="checkpoints", save_interval=1800,
                tokenizer_vocab=None, hidden_size=544, num_layers=10, dropout=0.1,
                resume_path=None):
    """
    ΕΚΠΑΙΔΕΥΣΗ του μοντέλου
    
    Αλγόριθμος:
    1. Παίρνουμε ένα batch από δεδομένα
    2. Το μοντέλο προβλέπει την επόμενη λέξη
    3. Συγκρίνουμε πρόβλεψη με πραγματικότητα (loss)
    4. Υπολογίζουμε gradients (backward)
    5. Ενημερώνουμε βάρη (optimizer step)
    6. Επαναλαμβάνουμε για όλα τα batches
    7. Save checkpoint κάθε save_interval δευτερόλεπτα
    
    Παράμετροι:
        model: το μοντέλο RWKV
        dataloader: φορτωτής δεδομένων (δίνει batches)
        epochs: πόσες φορές θα δει όλο το dataset
        lr: ρυθμός εκμάθησης (learning rate)
        device: 'cpu' ή 'cuda'
        checkpoint_dir: φάκελος για τα checkpoints
        save_interval: κάθε πόσα δευτερόλεπτα να γίνεται save (1800 = 30 λεπτά)
        resume_path: αν υπάρχει, φορτώνει checkpoint και συνεχίζει
    """
    model.to(device)
    
    # Adam optimizer: weight decay MONO σε μεγάλα projections (όχι σε biases/LayerNorm/etc)
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if 'weight' in name and p.ndim >= 2:
            decay_params.append(p)        # Linear/Embedding weights → decay
        else:
            no_decay_params.append(p)     # biases, LayerNorm, time_mix params → no decay
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr)
    
    # Loss function — μετράει πόσο απέχει η πρόβλεψη από την πραγματικότητα
    # Cross-entropy: όσο πιο κοντά στο 0, τόσο καλύτερα
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Αγνοούμε PAD tokens
    
    # Δημιουργία φακέλου για checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_epoch = 0
    skip_batches = 0
    total_batches = len(dataloader) * epochs
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_batches, eta_min=1e-5
    )
    
    # Αν υπάρχει resume_path, φορτώνουμε checkpoint
    resume_loss = None
    if resume_path and os.path.exists(resume_path):
        print(f"\nLoading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)  # 0-indexed epoch
        skip_batches = ckpt.get('batch', 0)
        resume_loss = ckpt.get('loss', None)
        if resume_loss is not None:
            print(f"Resuming from epoch {start_epoch+1}/{epochs}, batch {skip_batches} (loss: {resume_loss:.4f})")
        else:
            print(f"Resuming from epoch {start_epoch+1}/{epochs}, batch {skip_batches}")
    
    current_lr = scheduler.get_last_lr()[0]
    
    print(f"Training on {device}")
    print(f"Epochs: {epochs}, Learning rate: {lr} -> cosine decay to 1e-5")
    print(f"Weight decay: 0.1, Gradient clip: 1.0")
    print(f"Model dropout: {model.dropout}")
    print(f"Checkpoints every {save_interval//60} min -> {checkpoint_dir}/")
    if resume_path:
        print(f"Resumed at epoch {start_epoch+1}/{epochs} (batch {skip_batches}, LR: {current_lr:.6f})")
    
    # Live system monitor — updates at bottom-right via Win32/ANSI
    monitor = SystemMonitor()
    print("-" * 50)
    
    model.train()  # Λειτουργία εκπαίδευσης
    
    last_save_time = time.time()
    best_loss = resume_loss if resume_loss is not None else float('inf')
    
    # Χειρισμός Ctrl+C: αποθήκευση checkpoint πριν κλείσει
    import signal
    save_on_exit = {'should_save': False}
    
    def handle_sigint(sig, frame):
        print("\n\nInterrupt received! Saving checkpoint before exit...")
        save_on_exit['should_save'] = True
        # Αφήνουμε το τρέχον batch να τελειώσει
    
    signal.signal(signal.SIGINT, handle_sigint)
    
    for epoch in range(start_epoch, epochs):
        if save_on_exit['should_save']:
            break
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            if save_on_exit['should_save']:
                break
            
            # Resume: skip os batches που ήδη επεξεργαστήκαμε
            if epoch == start_epoch and batch_idx < skip_batches:
                continue
            
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Εμπρόσθια διάδοση: το μοντέλο προβλέπει
            logits, _ = model(input_ids)
            # logits: (batch, μήκος, vocab_size)
            
            # Υπολογισμός loss (σύγκριση πρόβλεψης με πραγματικότητα)
            # Αλλάζουμε σχήμα για το CrossEntropyLoss
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), target_ids.reshape(B * T))
            
            # Οπισθοδιάδοση: υπολογίζουμε gradients
            optimizer.zero_grad()   # Μηδενίζουμε προηγούμενα gradients
            loss.backward()         # Υπολογίζουμε νέα gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Κόβουμε ακραίες τιμές
            optimizer.step()        # Ενημερώνουμε βάρη
            scheduler.step()        # Μειώνουμε LR σταδιακά (cosine decay)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Live system monitor (CPU/RAM/temp) update
            monitor.update()
            
            # Εμφάνιση προόδου κάθε 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            
            # Save checkpoint κάθε save_interval δευτερόλεπτα
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                avg_loss = total_loss / max(num_batches, 1)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_e{epoch+1}_b{batch_idx+1}_loss{avg_loss:.3f}.pt")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'batch': batch_idx + 1,
                    'loss': avg_loss,
                    'vocab_size': len(tokenizer_vocab) if tokenizer_vocab else model.vocab_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'tokenizer_vocab': tokenizer_vocab,
                }, checkpoint_path)
                
                # Αν το loss είναι το καλύτερο, το κρατάμε και ως best
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocab_size': len(tokenizer_vocab) if tokenizer_vocab else model.vocab_size,
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'tokenizer_vocab': tokenizer_vocab,
                        'loss': avg_loss,
                        'epoch': epoch,
                        'batch': batch_idx + 1,
                    }, best_path)
                
                last_save_time = current_time
                print(f"  [OK] Checkpoint saved (loss: {avg_loss:.4f})")
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} completed | Average Loss: {avg_loss:.4f}")
        print("-" * 30)
    
    # Save checkpoint on exit (από Ctrl+C ή κανονικό τέλος)
    if save_on_exit['should_save']:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_interrupted.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'vocab_size': len(tokenizer_vocab) if tokenizer_vocab else model.vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'tokenizer_vocab': tokenizer_vocab,
            'loss': total_loss / max(num_batches, 1),
            'epoch': epoch,
            'batch': batch_idx + 1,
        }, checkpoint_path)
        print(f"  [OK] Interrupted checkpoint saved to {checkpoint_path}")
        print("Run again to resume from where you left off.")
    
    return model


# ---------- 3. MAIN — ΕΚΚΙΝΗΣΗ ΕΚΠΑΙΔΕΥΣΗΣ ----------

def main():
    # Ρυθμίσεις (μπορείς να τις αλλάξεις)
    DATASET_PATH = "aether_dataset.jsonl"
    MODEL_SAVE_PATH = "aether_model.pt"
    TOKENIZER_SAVE_PATH = "aether_tokenizer.json"
    
    BATCH_SIZE = 8        # Πόσα δείγματα ταυτόχρονα (8 βέλτιστο για CPU)
    MAX_LENGTH = 64       # Μέγιστο μήκος κειμένου (λέξεις)
    EPOCHS = 20           # Πόσες επαναλήψεις (~12-15 epochs σε 15 ώρες)
    LEARNING_RATE = 0.001 # Αρχικός ρυθμός εκμάθησης (cosine decay → 1e-5)
    HIDDEN_SIZE = 480     # Deep & narrow: 12×480 ~24.7M (vs 10×544 ~26.6M)
    NUM_LAYERS = 12       # Περισσότερα layers = καλύτερη γενίκευση
    DROPOUT = 0.1         # Dropout για πρόληψη overfitting
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # --- Βήμα 1: Φόρτωση dataset και χτίσιμο tokenizer ---
    print("\n=== Step 1: Building tokenizer ===")
    
    # Διαβάζουμε όλα τα κείμενα για να χτίσουμε το λεξιλόγιο
    texts = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
    
    # Δημιουργούμε tokenizer και χτίζουμε vocab
    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Αποθηκεύουμε τον tokenizer για μελλοντική χρήση
    tokenizer.save(TOKENIZER_SAVE_PATH)
    
    # --- Βήμα 2: Δημιουργία Dataset και DataLoader ---
    print("\n=== Step 2: Creating dataset ===")
    
    dataset = TextDataset(DATASET_PATH, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,      # Ανακάτεμα για καλύτερη εκμάθηση
        collate_fn=collate_batch,
        drop_last=True     # Αγνοούμε το τελευταίο ημιτελές batch
    )
    
    print(f"Batches per epoch: {len(dataloader)}")
    
    # --- Βήμα 3: Δημιουργία μοντέλου ---
    print("\n=== Step 3: Creating model ===")
    
    model = RWKV(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    # Συνολικές παράμετροι
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    
    # --- Βήμα 4: Εκπαίδευση ---
    print("\n=== Step 4: Training ===")
    
    # Έλεγχος για checkpoint — resume ή fresh start
    resume_path = None
    ckpt_dir = "checkpoints"
    ckpt_candidates = [
        os.path.join(ckpt_dir, "checkpoint_best.pt"),
        os.path.join(ckpt_dir, "checkpoint_interrupted.pt"),
    ]
    found = [c for c in ckpt_candidates if os.path.exists(c)]
    if found:
        print("Found existing checkpoint(s):")
        for i, c in enumerate(found):
            ckpt = torch.load(c, map_location='cpu')
            ep = ckpt.get('epoch', '?')
            loss = ckpt.get('loss', '?')
            print(f"  [{i+1}] {os.path.basename(c)} — epoch {ep}, loss: {loss}")
        print("  [R] Start fresh (delete checkpoints)")
        ans = input("Resume which? (1/2/R): ").strip().lower()
        if ans == 'r':
            for c in found:
                os.remove(c)
            print("Checkpoints deleted. Starting fresh.")
        elif ans in ('1', '2'):
            idx = int(ans) - 1
            resume_path = found[idx]
            print(f"Resuming from {os.path.basename(resume_path)}")
        else:
            print("Invalid choice. Starting fresh.")
    
    model = train_model(
        model=model,
        dataloader=dataloader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        checkpoint_dir=ckpt_dir,
        save_interval=1800,
        tokenizer_vocab=tokenizer.vocab,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        resume_path=resume_path,
    )
    
    # --- Βήμα 5: Αποθήκευση ---
    print("\n=== Step 5: Saving model ===")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'tokenizer_vocab': tokenizer.vocab,
    }, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # --- Βήμα 6: Δοκιμή ---
    print("\n=== Step 6: Quick test ===")
    
    test_prompt = "Hello"
    prompt_ids = tokenizer.encode(test_prompt, add_bos=True, add_eos=False)
    
    generated_ids = model.generate(
        prompt_ids,
        max_new=20,
        temperature=0.8,
        top_k=20
    )
    
    generated_text = tokenizer.decode(generated_ids, skip_special=True)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}")
    
    print("\n=== Training complete! ===")


if __name__ == "__main__":
    main()
