import torch
import json
import os
import sys
import argparse
import math
import time
from collections import deque

from tokenizer import ByteBPETokenizer as Tokenizer
from model import RWKV


# ─── Live Stream Visualizer with Neural + Math Display ──────────────────── #

class StreamVisualizer:
    """Live streaming visualizer: layer activity, token-by-token output, math sidebar."""

    def __init__(self, num_layers=12, tokenizer=None, temperature=0.7, top_k=40, rep_penalty=1.1):
        self.num_layers = num_layers
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_k = top_k
        self.rep_penalty = rep_penalty
        self._use_ansi = self._init_ansi()
        self._prompt_len = 0
        self._generated_text = ""
        self._start_time = 0
        self._prev_text = ""
        self._step = 0
        self._width = 72
        self._header_h = 6  # fixed header height (5 rows + 1 separator)

    @staticmethod
    def _init_ansi():
        if not sys.stdout.isatty():
            return False
        if os.name == 'nt':
            if os.environ.get('WT_SESSION') or os.environ.get('TERM_PROGRAM'):
                return True
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                return bool(kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7))
            except Exception:
                pass
            try:
                import colorama
                colorama.init()
                return True
            except ImportError:
                return False
        return True

    def _cl(self):
        return '\033[2K\r' if self._use_ansi else '\r' + ' ' * 120 + '\r'

    def _up(self, n):
        return f'\033[{n}A' if self._use_ansi else ''

    def start(self, prompt_len):
        self._prompt_len = prompt_len
        self._generated_text = ""
        self._prev_text = ""
        self._start_time = time.time()
        self._step = 0
        if self._use_ansi:
            self._draw_all(0, 1, [], None, None, 0)
        else:
            print("Aether: ", end="", flush=True)

    def _build_header(self, token_idx, total, activations, probs, speed):
        w = self._width
        pct = token_idx / max(total, 1)
        bar_len = 10
        filled = int(bar_len * pct)
        pbar = '█' * filled + '░' * (bar_len - filled)

        # Line 1: title + progress
        l1 = f"┌─ AETHER ── Neural Layers {'─'*10} Tok {token_idx:3d}/{total} {pbar} {pct*100:3.0f}% {'─'*5}┐"

        # Line 2: layer activity bars
        if activations and len(activations) >= self.num_layers:
            max_act = max(activations) if max(activations) > 0 else 1
            cells = []
            for a in activations[:self.num_layers]:
                r = a / max_act
                if r > 0.75:  cells.append('██')
                elif r > 0.5: cells.append('▓▓')
                elif r > 0.25: cells.append('▒▒')
                else:          cells.append('░░')
            layer_str = ' '.join(cells)
        else:
            layer_str = ' ░░' * self.num_layers
        l2 = f"│ Layers:  {layer_str}{' '*(w - 13 - len(layer_str))}│"

        # Line 3: pulse position
        pulse_pos = (token_idx - 1) % self.num_layers if token_idx > 0 else 0
        path_parts = []
        for i in range(self.num_layers):
            if i == pulse_pos:
                path_parts.append('●')
            elif abs(i - pulse_pos) <= 2:
                path_parts.append('○')
            else:
                path_parts.append('·')
        pulse_str = ' '.join(path_parts)
        tag = f"Emb {pulse_str} Head"
        l3 = f"│ Pulse:   {tag}{' '*(w - 13 - len(tag))}│"

        # Line 4: top-5 tokens (mini sidebar)
        if probs is not None:
            top_vals, top_idx = torch.topk(probs, min(5, probs.size(-1)))
            top_strs = [f"{self.tokenizer.decode([int(tid)], skip_special=True)} ({tv:.2f})"
                       for tid, tv in zip(top_idx, top_vals)][:5]
            top_line = ' · '.join(top_strs)
        else:
            top_line = ""
        top_line = top_line[:w-4]
        l4 = f"│ Top: {top_line:<{w-8}}│"

        # Line 5: stats
        elapsed = time.time() - self._start_time
        tok_s = (token_idx) / elapsed if elapsed > 0 else 0

        if probs is not None:
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        else:
            entropy = 0

        stats = f"H={entropy:.2f} · T={self.temperature:.2f} · K={self.top_k} · R={self.rep_penalty:.2f} · {tok_s:.1f} tok/s"
        l5 = f"│ {stats:<{w-4}}│"

        # Line 6: separator
        l6 = f"├{'─'*w}┤"

        return [l1, l2, l3, l4, l5, l6]

    def _draw_all(self, token_idx, total, activations, probs, speed, text_height):
        """Draw the full display: header + streaming text."""
        header = self._build_header(token_idx, total, activations, probs, speed)
        total_h = self._header_h + text_height + 1  # header + text + bottom border

        display = list(header)

        # Streaming text area
        if self._generated_text:
            display.append(f"│ {self._generated_text.replace(chr(10),' ')[:self._width]:<{self._width}} │")
        else:
            display.append(f"│ {'':<{self._width}} │")
        for _ in range(text_height - 1):
            display.append(f"│ {'':<{self._width}} │")

        display.append(f"└{'─'*self._width}┘")

        # Go up and redraw
        if hasattr(self, '_prev_h') and self._prev_h:
            sys.stdout.write(self._up(self._prev_h))
        for line in display:
            sys.stdout.write(self._cl() + line + '\n')
        sys.stdout.flush()
        self._prev_h = total_h

    def update(self, token_idx, total, activations, ids, logits=None, probs=None):
        self._step = token_idx
        new_ids = ids[self._prompt_len:]
        if self.tokenizer and new_ids:
            self._generated_text = self.tokenizer.decode(new_ids, skip_special=True)
            # Clean
            self._generated_text = self._generated_text.replace('<EOS>', '')

        if not self._use_ansi:
            sys.stdout.write(f"\rAether: {self._generated_text}")
            sys.stdout.flush()
            return

        # Compute text height
        text = self._generated_text.replace('\n', ' ')
        text_h = max(1, (len(text) // self._width) + 1)
        text_h = min(text_h, 8)  # max 8 lines

        elapsed = time.time() - self._start_time
        speed = token_idx / elapsed if elapsed > 0 else 0

        self._draw_all(token_idx, total, activations, probs, speed, text_h)

    def finish(self):
        if not self._use_ansi:
            print()
            return
        # Erase header, keep just the text
        self._generated_text = self._generated_text.replace('<EOS>', '').strip()
        # Go up to top of display and clear everything
        if hasattr(self, '_prev_h') and self._prev_h:
            sys.stdout.write(self._up(self._prev_h))
        # Print just the response without box
        sys.stdout.write(self._cl() + f"Aether: {self._generated_text}\n")
        sys.stdout.flush()


# ─── Model loading ───────────────────────────────────────────────────────── #

def find_best_checkpoint():
    import glob
    checkpoints = sorted(glob.glob("checkpoints/*.pt"))
    if not checkpoints:
        return None
    best = [c for c in checkpoints if "best" in c]
    if best:
        return best[0]
    return checkpoints[-1]


def load_model(model_path=None, tokenizer_path="aether_tokenizer.json", device='cpu'):
    if model_path is None or not os.path.exists(model_path):
        model_path = "checkpoints/checkpoint_best.pt"
    if not os.path.exists(model_path):
        model_path = "aether_model.pt"
    if not os.path.exists(model_path):
        model_path = find_best_checkpoint()
    if model_path is None or not os.path.exists(model_path):
        return None, None, None
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load tokenizer
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(vocab_size=8192)
        tokenizer.load(tokenizer_path)
    else:
        tokenizer = Tokenizer(vocab_size=checkpoint.get('vocab_size', 8192))

    model = RWKV(
        vocab_size=checkpoint.get('vocab_size', tokenizer.get_vocab_size()),
        hidden_size=checkpoint.get('hidden_size', 768),
        num_layers=checkpoint.get('num_layers', 12),
        head_size=checkpoint.get('head_size', 64),
        dropout=checkpoint.get('dropout', 0.1)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    print(f"Vocab size: {checkpoint.get('vocab_size', '?')}")
    print(f"Hidden size: {checkpoint.get('hidden_size', '?')}")
    print(f"Layers: {checkpoint.get('num_layers', '?')}")

    return model, tokenizer


# ─── Chat with visualization ─────────────────────────────────────────────── #

def chat(model, tokenizer, max_tokens=100, temperature=0.7, top_k=30,
         repetition_penalty=1.1, device='cpu', visual=True):
    print("\n" + "=" * 60)
    print("\n" + "=" * 60)
    print("  AETHER — Chat with your AI")
    print("  Type 'quit' to exit, 'reset' to start over")
    print("=" * 60)

    SEED = None
    history = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Aether: Goodbye! Thanks for chatting!")
            break

        if user_input.lower() == 'reset':
            history = []
            print("Aether: Conversation reset. Starting fresh!")
            continue

        if not user_input:
            continue

        history.append(f"User: {user_input}")

        prompt = "\n\n".join(history) + "\n\nAether:"
        stop_seq_ids = tokenizer.encode("User:", add_bos=False, add_eos=False)

        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        plen = len(prompt_ids)

        if visual:
            viz = StreamVisualizer(
                num_layers=model.num_layers,
                tokenizer=tokenizer,
                temperature=temperature,
                top_k=top_k,
                rep_penalty=repetition_penalty,
            )
            print()
            viz.start(plen)

            def progress_cb(tok_idx, total, activations, ids, logits, probs):
                viz.update(tok_idx, total, activations, ids, logits, probs)

            generated_ids = model.generate(
                prompt_ids,
                max_new=max_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                progress_cb=progress_cb,
                stop_seq_ids=stop_seq_ids
            )
            viz.finish()
            response = viz._generated_text
        else:
            print("Aether: ", end="", flush=True)
            generated_ids = model.generate(
                prompt_ids,
                max_new=max_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop_seq_ids=stop_seq_ids
            )
            new_ids = generated_ids[len(prompt_ids):]
            response = tokenizer.decode(new_ids, skip_special=True)

        cut_pos = response.find("User:")
        if cut_pos != -1:
            response = response[:cut_pos].rstrip()

        response = response.replace("<EOS>", "").strip()
        response = _clean_response(response)

        retries = 0
        t_retry = temperature
        while _is_hallucinated(response) and retries < 3 and t_retry > 0.1:
            retries += 1
            t_retry = max(t_retry * 0.5, 0.1)
            generated_ids = model.generate(prompt_ids, max_new=max_tokens, temperature=t_retry, top_k=top_k, repetition_penalty=repetition_penalty, stop_seq_ids=stop_seq_ids)
            new_ids = generated_ids[len(prompt_ids):]
            response = tokenizer.decode(new_ids, skip_special=True)
            cut_pos = response.find("User:")
            if cut_pos != -1:
                response = response[:cut_pos].rstrip()
            response = response.replace("<EOS>", "").strip()
            response = _clean_response(response)

        if _is_hallucinated(response):
            response = "I'm still learning. Ask me something else!"

        emoji, clean = _format_with_emoji(response)
        display = f"{emoji} {clean}".strip() if emoji else clean
        if not visual:
            print(display)
        else:
            print(f"Aether: {display}")

        history.append(f"Aether: {response}")

        if len(history) > 20:
            history = history[-20:]


def _format_with_emoji(text):
    import re
    m = re.match(r'<(\w+)>\s*(.*)', text.strip())
    if not m:
        return '', text
    tag = m.group(1)
    rest = m.group(2)
    emojis = {
        "joy": "[joy]", "sadness": "[sad]", "anger": "[anger]", "fear": "[fear]",
        "surprise": "[wow]", "confusion": "[hmm]", "neutral": "",
        "sarcasm": "[sarcasm]", "encouragement": "[keep going]", "curiosity": "[?]",
    }
    prefix = emojis.get(tag, "")
    return prefix, rest


# ─── Quick test ──────────────────────────────────────────────────────────── #

def _is_hallucinated(response):
    r = response.strip()
    if len(r) < 10:
        return True
    words = r.split()
    if len(words) >= 4 and len(set(words)) <= 2:
        return True
    bad = ["can't help", "can't write", "can't know", "not know", "can't make",
           "can't do", "don't know", "can't tell", "can't answer", "let me",
           "can only", "can't say", "can't give", "can't think", "can't find",
           "can't remember"]
    for b in bad:
        if b in r.lower():
            return True
    return False


def _clean_response(response):
    r = response.strip()
    r = r.replace("Aether:", "").strip()
    cut_at = ["Let me", "and the ", "and a ", "and can", "I can't", "not know",
              "can't help", "can't write", "can't make", "can't do", "can only"]
    for c in cut_at:
        idx = r.lower().find(c.lower())
        if idx >= 0:
            r = r[:idx].rstrip()
    for end in [". ", "! ", "? "]:
        idx = r.rfind(end)
        if idx > 0:
            r = r[:idx+1]
    return r.strip()


def test_generation(model, tokenizer, prompt="User: Hello\n\nAether:",
                    max_tokens=30, temperature=0.7, top_k=30,
                    repetition_penalty=1.1, device='cpu', visual=True):
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    stop_seq_ids = tokenizer.encode("User:", add_bos=False, add_eos=False)

    if visual:
        viz = StreamVisualizer(
            num_layers=model.num_layers,
            tokenizer=tokenizer,
            temperature=temperature,
            top_k=top_k,
            rep_penalty=repetition_penalty,
        )
        print()
        viz.start(len(prompt_ids))

        def progress_cb(tok_idx, total, activations, ids, logits, probs):
            viz.update(tok_idx, total, activations, ids, logits, probs)

        generated_ids = model.generate(
            prompt_ids,
            max_new=max_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            progress_cb=progress_cb,
            stop_seq_ids=stop_seq_ids
        )
        viz.finish()
        generated_text = viz._generated_text
    else:
        generated_ids = model.generate(
            prompt_ids,
            max_new=max_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_seq_ids=stop_seq_ids
        )
        new_ids = generated_ids[len(prompt_ids):]
        generated_text = tokenizer.decode(new_ids, skip_special=True)

    cut_pos = generated_text.find("User:")
    if cut_pos != -1:
        generated_text = generated_text[:cut_pos].rstrip()

    generated_text = generated_text.replace("<EOS>", "").strip()
    generated_text = _clean_response(generated_text)

    retries = 0
    t_retry = temperature
    while _is_hallucinated(generated_text) and retries < 3 and t_retry > 0.1:
        retries += 1
        t_retry = max(t_retry * 0.5, 0.1)
        generated_ids = model.generate(prompt_ids, max_new=max_tokens, temperature=t_retry, top_k=top_k, repetition_penalty=repetition_penalty, stop_seq_ids=stop_seq_ids)
        new_ids = generated_ids[len(prompt_ids):]
        generated_text = tokenizer.decode(new_ids, skip_special=True)
        cut_pos = generated_text.find("User:")
        if cut_pos != -1:
            generated_text = generated_text[:cut_pos].rstrip()
        generated_text = generated_text.replace("<EOS>", "").strip()
        generated_text = _clean_response(generated_text)

    if _is_hallucinated(generated_text):
        generated_text = "I'm still learning. Ask me something else!"

    emoji, clean = _format_with_emoji(generated_text)
    display = f"{emoji} {clean}".strip() if emoji else clean
    if not visual:
        print(f"\nYou: {prompt.replace('User: ','').replace('\n\nAether:','')}")
    print(f"Aether: {display}")

    return generated_text


# ─── Main ────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Aether - Chat with your AI")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature (0=deterministic, 1=random)")
    parser.add_argument("--topk", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--rep", type=float, default=1.1, help="Repetition penalty (>1.0 reduces repetition)")
    parser.add_argument("--maxtokens", type=int, default=80, help="Maximum response length")
    parser.add_argument("--quick", type=str, default=None, help="Single-turn test (e.g. --quick 'Who are you?')")
    parser.add_argument("--novis", action="store_true", help="Disable live visualization")
    args = parser.parse_args()

    if os.path.exists("checkpoints/checkpoint_best.pt"):
        MODEL_PATH = "checkpoints/checkpoint_best.pt"
    elif os.path.exists("aether_model.pt"):
        MODEL_PATH = "aether_model.pt"
    else:
        MODEL_PATH = find_best_checkpoint()
        if MODEL_PATH is None:
            MODEL_PATH = "aether_model.pt"
    TOKENIZER_PATH = "aether_tokenizer.json"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(MODEL_PATH) and not os.path.exists("aether_model.pt"):
        print(f"ERROR: No model file found!")
        print("Run train.py first to train the model.")
        return

    if not os.path.exists(TOKENIZER_PATH) and not os.path.exists("checkpoints"):
        print(f"ERROR: Tokenizer file '{TOKENIZER_PATH}' not found!")
        print("Run train.py first to build the tokenizer.")
        return

    model, tokenizer = load_model(MODEL_PATH, TOKENIZER_PATH, device)

    if args.quick:
        prompt = f"User: {args.quick}\n\nAether:"
        test_generation(model, tokenizer, prompt, max_tokens=args.maxtokens,
                        temperature=args.temp, top_k=args.topk,
                        repetition_penalty=args.rep, device=device,
                        visual=not args.novis)
        return

    print(f"\nSettings: temp={args.temp}, top_k={args.topk}, rep_penalty={args.rep}")
    chat(model, tokenizer, max_tokens=args.maxtokens, temperature=args.temp,
         top_k=args.topk, repetition_penalty=args.rep, device=device,
         visual=not args.novis)


if __name__ == "__main__":
    main()
