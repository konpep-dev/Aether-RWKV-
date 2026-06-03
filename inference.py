import torch
import json
import os
import sys
import argparse
import math

from tokenizer import Tokenizer
from model import RWKV


# ─── Live Neural Network Visualizer ──────────────────────────────────────── #

class NeuralVisualizer:
    """Live visualization of the neural network during generation."""

    def __init__(self, num_layers=10, width=56):
        self.num_layers = num_layers
        self.width = width
        self.tokenizer = None
        self._text = ''
        self._use_ansi = self._init_ansi()
        self._lines = self._build_initial()
        self._fired = False
        self._decoded = ''

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

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _cl(self):
        """Clear current line and return to start."""
        if self._use_ansi:
            return '\033[2K\r'
        return '\r' + ' ' * 120 + '\r'

    def _up(self, n):
        return f'\033[{n}A' if self._use_ansi else ''

    def _build_initial(self):
        w = self.width
        lines = []
        lines.append(f'+--- AETHER --- Token     0/???   0% ---+')
        lines.append(f'| Layers: .. .. .. .. .. .. .. .. .. ..  |')
        lines.append(f'| Pulse:  Embed -@- - - - - - - - Head  |')
        lines.append(f'|{"":56s}|')
        lines.append(f'+{"-"*58}+')
        return lines

    def start(self):
        self._fired = False
        self._decoded = ''
        self._text = ''
        for l in self._lines:
            print(l)
        self._box_height = len(self._lines)

    def _redraw(self, lines):
        if self._use_ansi:
            out = self._up(self._box_height)
            for l in lines:
                out += self._cl() + l + '\n'
            sys.stdout.write(out)
            sys.stdout.flush()
        else:
            for l in lines:
                sys.stdout.write(self._cl() + l + '\n')
            sys.stdout.flush()

    def update(self, token_idx, total_tokens, activations, token_ids):
        bar_len = 12
        pct = token_idx / max(total_tokens, 1)
        filled = int(bar_len * pct)
        bar = '#' * filled + '.' * (bar_len - filled)

        if activations and len(activations) >= self.num_layers:
            max_act = max(activations) if max(activations) > 0 else 1
            cell = []
            for a in activations[:self.num_layers]:
                r = a / max_act
                if r > 0.75:  cell.append('##')
                elif r > 0.5: cell.append('%%')
                elif r > 0.25: cell.append('==')
                else:          cell.append('..')
            layer_str = ' '.join(cell)
        else:
            layer_str = '.. .. .. .. .. .. .. .. .. ..'

        pulse_pos = (token_idx - 1) % self.num_layers if token_idx > 0 else 0
        path = ['@' if i == pulse_pos else '-' for i in range(self.num_layers)]
        pulse = ' '.join(path)

        if self.tokenizer and len(token_ids) > 0:
            self._decoded = self.tokenizer.decode(token_ids, skip_special=True)

        show = (self._decoded[-50:] if len(self._decoded) > 50 else self._decoded).replace('\n', ' ')

        pct_str = f'{pct*100:3.0f}%'
        l1 = f'+--- AETHER --- Token {token_idx:3d}/{total_tokens} [{bar}] {pct_str} ---+'
        l2 = f'| Layers: {layer_str}  |'
        l3 = f'| Pulse:  Embed {pulse} Head  |'
        l4 = f'| {show:<56s}|'
        l5 = f'+{"-"*58}+'

        self._redraw([l1, l2, l3, l4, l5])

    def finish(self):
        final = self._decoded.replace('<EOS>', '').strip()
        l4 = f'| {final:<56s}|'
        l5 = f'+{"-"*58}+'
        self._redraw([self._lines[0], self._lines[1], self._lines[2], l4, l5])
        print()


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

    tokenizer = Tokenizer(vocab=checkpoint['tokenizer_vocab'])

    model = RWKV(
        vocab_size=checkpoint['vocab_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    print(f"Vocab size: {checkpoint['vocab_size']}")
    print(f"Hidden size: {checkpoint['hidden_size']}")
    print(f"Layers: {checkpoint['num_layers']}")

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
    viz = NeuralVisualizer(num_layers=model.num_layers) if visual else None
    if viz:
        viz.set_tokenizer(tokenizer)

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
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        plen = len(prompt_ids)

        if viz:
            def progress_cb(tok_idx, total, activations, ids):
                viz.update(tok_idx, total, activations, ids[plen:])

            print()
            viz.start()

            generated_ids = model.generate(
                prompt_ids,
                max_new=max_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                progress_cb=progress_cb
            )
            viz.finish()
        else:
            print("Aether: ", end="", flush=True)
            generated_ids = model.generate(
                prompt_ids,
                max_new=max_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty
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
            generated_ids = model.generate(prompt_ids, max_new=max_tokens, temperature=t_retry, top_k=top_k, repetition_penalty=repetition_penalty)
            new_ids = generated_ids[len(prompt_ids):]
            response = tokenizer.decode(new_ids, skip_special=True)
            cut_pos = response.find("User:")
            if cut_pos != -1:
                response = response[:cut_pos].rstrip()
            response = response.replace("<EOS>", "").strip()
            response = _clean_response(response)

        if _is_hallucinated(response):
            response = "I'm still learning. Ask me something else!"

        if not viz:
            print(response)
        else:
            print(f"Aether: {response}")

        history.append(f"Aether: {response}")

        if len(history) > 20:
            history = history[-20:]


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

    viz = NeuralVisualizer(num_layers=model.num_layers) if visual else None
    if viz:
        viz.set_tokenizer(tokenizer)
        plen = len(prompt_ids)

        def progress_cb(tok_idx, total, activations, ids):
            viz.update(tok_idx, total, activations, ids[plen:])

        print()
        viz.start()

        generated_ids = model.generate(
            prompt_ids,
            max_new=max_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            progress_cb=progress_cb
        )
        viz.finish()
    else:
        generated_ids = model.generate(
            prompt_ids,
            max_new=max_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty
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
        generated_ids = model.generate(prompt_ids, max_new=max_tokens, temperature=t_retry, top_k=top_k, repetition_penalty=repetition_penalty)
        new_ids = generated_ids[len(prompt_ids):]
        generated_text = tokenizer.decode(new_ids, skip_special=True)
        cut_pos = generated_text.find("User:")
        if cut_pos != -1:
            generated_text = generated_text[:cut_pos].rstrip()
        generated_text = generated_text.replace("<EOS>", "").strip()
        generated_text = _clean_response(generated_text)

    if _is_hallucinated(generated_text):
        generated_text = "I'm still learning. Ask me something else!"

    if not visual:
        print(f"\nYou: {prompt.replace('User: ','').replace('\n\nAether:','')}")
    print(f"Aether: {generated_text}")

    return generated_text


# ─── Main ────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Aether - Chat with your AI")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature (0=σταθερό, 1=τυχαίο)")
    parser.add_argument("--topk", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--rep", type=float, default=1.3, help="Repetition penalty (>1.0 μειώνει επαναλήψεις)")
    parser.add_argument("--maxtokens", type=int, default=80, help="Μέγιστο μήκος απάντησης")
    parser.add_argument("--quick", type=str, default=None, help="Quick test (π.χ. --quick 'Who are you?')")
    parser.add_argument("--novis", action="store_true", help="Απενεργοποίηση live visualization")
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
