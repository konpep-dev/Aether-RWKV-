"""
RWKV v4 — Σωστή υλοποίηση με time-mix WKV formula

Αλλαγές από προηγούμενη version:
1. Σωστό time-mix WKV με time_decay + time_first (bonus)
2. state_num = w * state_num + exp(k) * v  (κρατάει μόνο το παρελθόν)
3. wkv_t = (state_num + exp(u) * exp(k_t) * v_t) / (state_den + exp(u) * exp(k_t))
4. ~25M params (hidden=544, layers=10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RWKV_TimeMix(nn.Module):
    """
    Time-Mix με σωστό WKV formula
    
    wkv_t = (sum_{past} w * exp(k_i) * v_i + exp(u) * exp(k_t) * v_t) /
            (sum_{past} w * exp(k_i) + exp(u) * exp(k_t))
    
    όπου w = exp(-exp(time_decay)), u = time_first
    """
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # time_decay: πόσο γρήγορα ξεχνάει το παρελθόν (ανά κανάλι)
        # initialized στο log(1) ≈ 0, οπότε exp(-exp(0)) ≈ 0.37
        self.time_decay = nn.Parameter(torch.empty(hidden_size))
        # time_first: bonus στο τρέχον token (ανά κανάλι)
        self.time_first = nn.Parameter(torch.empty(hidden_size))
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.orthogonal_(self.key.weight, gain=0.1)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.orthogonal_(self.receptance.weight, gain=1.0)
        nn.init.constant_(self.time_decay, 0.0)
        nn.init.constant_(self.time_first, 5.0)
    
    def forward(self, x, state=None):
        """
        Time-mix για εκπαίδευση (παράλληλη επεξεργασία)
        
        Υπολογίζει σωστό WKV:
        - state_num = w * state_num + exp(k) * v  (μόνο παρελθόν)
        - num = w * state_num + exp(u) * exp(k_t) * v_t  (παρελθόν + bonus τρέχον)
        """
        B, T, C = x.shape
        
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        
        decay = torch.exp(-torch.exp(self.time_decay)).view(1, 1, C)
        exp_bonus = torch.exp(self.time_first).view(1, 1, C)
        
        exp_k = torch.exp(k)
        
        num = torch.zeros(B, T, C, device=x.device)
        den = torch.zeros(B, T, C, device=x.device)
        
        if state is not None:
            s_num, s_den = state
        else:
            s_num = torch.zeros(B, 1, C, device=x.device)
            s_den = torch.zeros(B, 1, C, device=x.device)
        
        for t in range(T):
            ek = exp_k[:, t:t+1, :]
            v_t = v[:, t:t+1, :]
            
            num_t = s_num * decay + exp_bonus * ek * v_t
            den_t = s_den * decay + exp_bonus * ek
            num[:, t:t+1, :] = num_t
            den[:, t:t+1, :] = den_t
            
            s_num = s_num * decay + ek * v_t
            s_den = s_den * decay + ek
        
        wkv = num / (den + 1e-8)
        output = torch.sigmoid(r) * wkv
        return output, (s_num, s_den)
    
    def forward_step(self, x, state):
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        
        decay = torch.exp(-torch.exp(self.time_decay)).view(1, -1, 1)
        exp_bonus = torch.exp(self.time_first).view(1, -1, 1)
        
        s_num, s_den = state
        ek = torch.exp(k)
        
        num = s_num * decay + exp_bonus * ek * v
        den = s_den * decay + exp_bonus * ek
        wkv = num / (den + 1e-8)
        
        s_num = s_num * decay + ek * v
        s_den = s_den * decay + ek
        
        output = torch.sigmoid(r) * wkv
        return output, (s_num, s_den)


class RWKV_ChannelMix(nn.Module):
    """
    Channel-Mix: MLP με ReLU² + Sigmoid gate
    
    Δομή: Linear(×2) → ReLU² → Linear(÷2) × Sigmoid(receptance)
    """
    
    def __init__(self, hidden_size):
        super().__init__()
        ff_size = hidden_size * 2
        
        self.key = nn.Linear(hidden_size, ff_size, bias=False)
        self.value = nn.Linear(ff_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.orthogonal_(self.key.weight, gain=1.0)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.receptance.weight)
    
    def forward(self, x):
        k = F.relu(self.key(x)) ** 2
        v = self.value(k)
        r = torch.sigmoid(self.receptance(x))
        return r * v


class RWKV_Block(nn.Module):
    """
    Block: LayerNorm → TimeMix → Residual → LayerNorm → ChannelMix → Residual
    """
    
    def __init__(self, hidden_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.time_mix = RWKV_TimeMix(hidden_size)
        self.channel_mix = RWKV_ChannelMix(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, state=None):
        tm_out, new_state = self.time_mix(self.ln1(x), state)
        x = x + self.dropout(tm_out)
        cm_out = self.channel_mix(self.ln2(x))
        x = x + self.dropout(cm_out)
        return x, new_state
    
    def forward_step(self, x, state):
        tm_out, tm_state = self.time_mix.forward_step(self.ln1(x), state[0])
        x = x + tm_out
        cm_out = self.channel_mix(self.ln2(x))
        x = x + cm_out
        return x, (tm_state,)


class RWKV(nn.Module):
    """
    Πλήρες μοντέλο RWKV v4
    
    Embedding → 10× RWKV_Block → LayerNorm → Linear(έξοδος)
    ~25M params (hidden=544)
    """
    
    def __init__(self, vocab_size, hidden_size=480, num_layers=12, ff_size_mult=2, dropout=0.0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Αλλάζουμε το ff_size αν χρειάζεται
        RWKV_ChannelMix._ff_mult = ff_size_mult
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            RWKV_Block(hidden_size, dropout) for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.embed.weight = self.head.weight
        
        self._block_outputs = None
        
        # Init head ειδικά: orthogonal gain = 0.5 * sqrt(vocab / hidden)
        head_gain = 0.5 * (vocab_size / hidden_size) ** 0.5
        nn.init.orthogonal_(self.head.weight, gain=head_gain)
        
        # Init υπόλοιπων layers (LayerNorm)
        nn.init.ones_(self.ln_out.weight)
        nn.init.zeros_(self.ln_out.bias)
        
        total = sum(p.numel() for p in self.parameters())
        print(f"RWKV (v4 WKV) created: {total:,} total parameters")
    
    def forward(self, x, state=None):
        x = self.embed_drop(self.embed(x))
        new_states = []
        for i, block in enumerate(self.blocks):
            block_state = state[i] if state is not None else None
            x, new_state = block(x, block_state)
            new_states.append(new_state)
        x = self.ln_out(x)
        logits = self.head(x)
        return logits, new_states
    
    def forward_step(self, x, state):
        x = self.embed(x)
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block.forward_step(x, state[i])
            new_states.append(new_state)
            if self._block_outputs is not None:
                self._block_outputs.append(x.abs().mean().item())
        x = self.ln_out(x)
        logits = self.head(x)
        return logits, new_states
    
    def init_state(self, batch_size=1, device='cpu'):
        state = []
        for block in self.blocks:
            n = torch.zeros(batch_size, 1, self.hidden_size, device=device)
            d = torch.zeros(batch_size, 1, self.hidden_size, device=device)
            state.append(((n, d),))
        return state
    
    def generate(self, prompt_ids, max_new=50, temperature=0.7, top_k=20, repetition_penalty=1.1, progress_cb=None):
        self.eval()
        device = next(self.parameters()).device
        generated = list(prompt_ids)
        state = self.init_state(device=device)
        
        with torch.no_grad():
            for token_id in prompt_ids[:-1]:
                x = torch.tensor([[token_id]], device=device)
                _, state = self.forward_step(x, state)
            
            x = torch.tensor([[prompt_ids[-1]]], device=device)
            
            for step in range(max_new):
                if progress_cb:
                    self._block_outputs = []
                
                logits, state = self.forward_step(x, state)
                
                if progress_cb:
                    progress_cb(step + 1, max_new, list(self._block_outputs), generated)
                    self._block_outputs = None
                
                if repetition_penalty > 1.0:
                    for tid in set(generated[-20:]):
                        if logits[tid] > 0:
                            logits[tid] /= repetition_penalty
                        else:
                            logits[tid] *= repetition_penalty
                
                logits = logits / temperature
                
                if top_k > 0:
                    vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < vals[-1]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_id)
                x = torch.tensor([[next_id]], device=device)
                
                if next_id == 2:
                    break
        
        self._block_outputs = None
        return generated
    
if __name__ == "__main__":
    model = RWKV(vocab_size=1024, hidden_size=128, num_layers=2)
    x = torch.randint(0, 1024, (2, 10))
    logits, state = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
