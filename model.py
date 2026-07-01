import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os

# Try to import optimized WKV kernel
try:
    from wkv_cuda_kernel import wkv_forward_fast, warmup_wkv_kernel
    USE_FAST_WKV = True
    print("[OK] Optimized WKV kernel available (TorchScript/Triton)")
except ImportError:
    USE_FAST_WKV = False
    print("[WARNING] Using fallback WKV (slower)")

class TimeMix(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.time_decay = nn.Parameter(torch.empty(hidden_size))
        self.time_first = nn.Parameter(torch.empty(hidden_size))
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.key.weight, gain=0.5)
        nn.init.orthogonal_(self.value.weight, gain=0.5)
        nn.init.orthogonal_(self.receptance.weight, gain=0.5)
        with torch.no_grad():
            self.time_decay.fill_(0)
            self.time_first.fill_(0)

    @staticmethod
    def _wkv_step(k_t, v_t, r_t, num_s, den_s, decay, first):
        ek = torch.exp(k_t)
        euk = torch.exp(first + k_t)
        wkv_num = num_s + euk * v_t
        wkv_den = den_s + euk
        wkv = wkv_num / (wkv_den + 1e-8)
        out_t = torch.sigmoid(r_t) * wkv
        new_num = decay * num_s + ek * v_t
        new_den = decay * den_s + ek
        return out_t, new_num, new_den

    def forward(self, x):
        B, T, C = x.shape
        r = self.receptance(x)
        k = self.key(x)
        v = self.value(x)
        
        # Use optimized CUDA kernel if available
        if USE_FAST_WKV:
            try:
                out = wkv_forward_fast(k, v, r, self.time_decay, self.time_first)
                return out.to(x.dtype)
            except Exception as e:
                # Fallback to Python if kernel fails
                print(f"[WARNING] CUDA kernel failed ({e}), using fallback")
        
        # Fallback: Simple sequential WKV - O(T) complexity
        w = -torch.exp(self.time_decay.float())  # [C] - log decay
        u = self.time_first.float()  # [C] - bonus
        
        k_f = k.float()  # [B, T, C]
        v_f = v.float()
        r_f = r.float()
        
        # Initialize output and states
        wkv_output = torch.empty(B, T, C, device=x.device, dtype=torch.float32)
        
        # Initialize state (numerator and denominator)
        num_state = torch.zeros(B, C, device=x.device, dtype=torch.float32)
        den_state = torch.zeros(B, C, device=x.device, dtype=torch.float32)
        decay = torch.exp(w)  # [C]
        
        # Sequential recurrence - unavoidable due to data dependency
        for t in range(T):
            ek = torch.exp(k_f[:, t, :])  # [B, C]
            euk = torch.exp(u + k_f[:, t, :])  # [B, C]
            
            # WKV formula
            wkv_num = num_state + euk * v_f[:, t, :]
            wkv_den = den_state + euk
            wkv_output[:, t, :] = wkv_num / (wkv_den + 1e-8)
            
            # Update state for next timestep
            num_state = decay * num_state + ek * v_f[:, t, :]
            den_state = decay * den_state + ek
        
        out = torch.sigmoid(r_f) * wkv_output
        return out.to(x.dtype)

    def forward_step(self, x, state):
        if x.dim() == 2:
            x = x.squeeze(0)
        xf = x.float()
        r = self.receptance(xf)
        k = self.key(xf)
        v = self.value(xf)
        decay = torch.exp(-torch.exp(self.time_decay.float()))
        first = self.time_first.float()

        ek = torch.exp(k)
        euk = torch.exp(first + k)

        num_s, den_s = state
        wkv_num = num_s + euk * v
        wkv_den = den_s + euk
        wkv = wkv_num / (wkv_den + 1e-8)
        out = torch.sigmoid(r) * wkv.to(x.dtype)

        num_s = decay * num_s + ek * v
        den_s = decay * den_s + ek
        return out, (num_s, den_s)

    def init_state(self, device, dtype):
        C = self.key.weight.size(1)
        return (
            torch.zeros(C, device=device, dtype=torch.float32),
            torch.zeros(C, device=device, dtype=torch.float32)
        )


class ChannelMix(nn.Module):
    def __init__(self, hidden_size, ff_hidden=None):
        super().__init__()
        if ff_hidden is None:
            ff_hidden = hidden_size * 2
        self.key = nn.Linear(hidden_size, ff_hidden, bias=False)
        self.value = nn.Linear(ff_hidden, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.key.weight, gain=0.5)
        nn.init.zeros_(self.value.weight)
        nn.init.orthogonal_(self.receptance.weight, gain=0.5)

    def forward(self, x):
        r = torch.sigmoid(self.receptance(x))
        k = torch.square(F.relu(self.key(x)))
        return r * self.value(k)

    def forward_step(self, x):
        if x.dim() == 2:
            x = x.squeeze(0)
        r = torch.sigmoid(self.receptance(x))
        k = torch.square(F.relu(self.key(x)))
        return r * self.value(k)


class Block(nn.Module):
    def __init__(self, hidden_size, layer_id, ff_hidden=None):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.time_mix = TimeMix(hidden_size)
        self.channel_mix = ChannelMix(hidden_size, ff_hidden)

    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x

    def forward_step(self, x, att_state):
        x_att = self.ln1(x)
        att_out, new_att_state = self.time_mix.forward_step(x_att, att_state)
        x = x + att_out
        x = x + self.channel_mix(self.ln2(x))
        return x, new_att_state


class RWKV(nn.Module):
    def __init__(self, vocab_size=5562, hidden_size=640, num_layers=14, head_size=None, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(hidden_size, i) for i in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        # Scale tied weights so initial logits are O(1), not O(sqrt(h))
        with torch.no_grad():
            self.embed.weight.div_(hidden_size ** 0.5)

        self._block_outputs = None
        total = sum(p.numel() for p in self.parameters())
        print(f"RWKV v4 created: {total:,} total parameters")
        print(f"  ~{total * 4 / 1024 / 1024:.0f}MB float32")

    def forward(self, x, state=None):
        x = self.embed_drop(self.embed(x))
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        logits = self.head(x)
        return logits, None

    def forward_step(self, x, state):
        # x: [1, 1] or [1] - single token
        x = self.embed(x)
        # Squeeze to [hidden_size] for recurrent processing
        if x.dim() == 3:
            x = x.squeeze(0).squeeze(0)  # [1,1,H] -> [H]
        elif x.dim() == 2:
            x = x.squeeze(0)  # [1,H] -> [H]
        
        new_states = []
        for i, block in enumerate(self.blocks):
            att_state = state[i]
            x, new_att_state = block.forward_step(x, att_state)
            new_states.append(new_att_state)
            if self._block_outputs is not None:
                self._block_outputs.append(x.abs().mean().item())
        x = self.ln_out(x)
        logits = self.head(x)  # [vocab_size]
        # Always return [1, vocab_size] for consistency
        return logits.unsqueeze(0), new_states

    def init_state(self, batch_size=1, device='cpu', dtype=None):
        if dtype is None:
            dtype = next(self.embed.parameters()).dtype
        state = []
        for block in self.blocks:
            state.append(block.time_mix.init_state(device, dtype))
        return state

    def generate(self, input_ids, max_new=50, temperature=0.8, top_k=40, repetition_penalty=1.0, stop_tokens=None):
        """
        Generate text autoregressively using the model.
        
        Args:
            input_ids: List of token IDs (prompt)
            max_new: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: List of token IDs that stop generation
        
        Returns:
            List of all token IDs (prompt + generated)
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # Initialize state
        state = self.init_state(batch_size=1, device=device, dtype=dtype)
        
        # Track generated tokens
        all_ids = list(input_ids)
        stop_tokens = stop_tokens or []
        
        # Process prompt through model to build state
        for i, token_id in enumerate(input_ids):
            inp = torch.tensor([[token_id]], device=device, dtype=torch.long)
            logits, state = self.forward_step(inp, state)
        
        # Generate new tokens
        for _ in range(max_new):
            # Get logits for next token - shape [1, vocab_size] -> [vocab_size]
            lgs = logits[0].clone()
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for prev_id in set(all_ids[-64:]):  # only last 64 tokens
                    if prev_id < len(lgs):
                        if lgs[prev_id] > 0:
                            lgs[prev_id] /= repetition_penalty
                        else:
                            lgs[prev_id] *= repetition_penalty
            
            # Apply temperature
            lgs = lgs / max(temperature, 1e-6)
            
            # Top-k sampling
            if top_k > 0:
                top_k_actual = min(top_k, lgs.size(-1))
                vals, _ = torch.topk(lgs, top_k_actual)
                lgs[lgs < vals[-1]] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(lgs, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            
            # Check stop condition
            if next_id in stop_tokens:
                break
            if next_id == 2:  # EOS token
                break
            
            all_ids.append(next_id)
            
            # Forward step for next iteration
            inp = torch.tensor([[next_id]], device=device, dtype=torch.long)
            logits, state = self.forward_step(inp, state)
        
        return all_ids

    def compile_model(self, mode="default", fullgraph=False):
        if os.name == 'nt':
            return False
        if hasattr(torch, "compile") and os.environ.get("RWKV_NO_COMPILE", "0") == "0":
            try:
                TimeMix._wkv_step = torch.compile(TimeMix._wkv_step, mode=mode, fullgraph=fullgraph)
                torch.set_float32_matmul_precision('high')
                return True
            except Exception:
                pass
        return False

    def optimize_for_cpu(self, n_threads=None):
        import multiprocessing
        if n_threads is None:
            n_threads = min(multiprocessing.cpu_count(), 8)
        torch.set_num_threads(n_threads)
        if torch.backends.mkldnn.is_available():
            torch.backends.mkldnn.enabled = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    model = RWKV(vocab_size=5562, hidden_size=640, num_layers=14)
    x = torch.randint(0, 5562, (1, 10))
    logits, _ = model(x)
    print(f"Output shape: {logits.shape}")
    state = model.init_state(device='cpu')
    for t in range(10):
        l, state = model.forward_step(x[:, t:t+1], state)
    print("Forward step OK")
