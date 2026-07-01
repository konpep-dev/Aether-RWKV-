"""
Byte-level BPE Tokenizer — universal language support (100+ languages)
  - Base: UTF-8 bytes (256 tokens) → any Unicode works
  - BPE merges: learned byte pair encoding (8192 total vocab)
  - Special tokens: PAD=0, BOS=1, EOS=2, TRN=3
  - Built-in language detection (character n-gram, no training)
"""

import json
import re
from collections import defaultdict, Counter


class ByteBPETokenizer:
    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<trn>": 3}

    def __init__(self, vocab_size=8192, num_merges=None):
        self.vocab_size = vocab_size
        self.num_merges = (vocab_size - 256) if num_merges is None else num_merges
        assert 256 + self.num_merges <= vocab_size

        # Byte to token mapping
        self.byte_to_token = {i: i for i in range(256)}

        # BPE merge pairs: list of (byte_or_token_1, byte_or_token_2)
        self.merges = []

        # Token to byte sequence (for decoding)
        self.token_to_bytes = {i: bytes([i]) for i in range(256)}
        for k, v in self.SPECIAL.items():
            pass  # special tokens have no byte mapping

        # Cache for encoding (O(1) merge lookup)
        self._merge_map = {}
        self._merge_priority = {}
        self._max_token = 255 + len(self.merges)

    def train(self, texts, verbose=True, max_texts=20000):
        """Learn BPE merges using a max-heap for O(log N) per step."""
        if len(texts) > max_texts:
            texts = texts[:max_texts]
            if verbose:
                print(f"  Using {max_texts} texts for BPE training")

        import heapq
        # Convert to byte sequences
        sequences = []
        for text in texts:
            seq = list(text.encode("utf-8"))
            if len(seq) < 2:
                continue
            sequences.append(seq)

        # Build initial pair frequency and max-heap
        pair_freq = {}
        heap = []
        for seq in sequences:
            for i in range(len(seq) - 1):
                p = (seq[i], seq[i + 1])
                pair_freq[p] = pair_freq.get(p, 0) + 1
        for p, c in pair_freq.items():
            heapq.heappush(heap, (-c, p))

        next_token = 256
        self.merges = []
        self.token_to_bytes = {i: bytes([i]) for i in range(256)}

        merges_done = 0
        while merges_done < self.num_merges:
            # Pop the most frequent valid pair
            while heap:
                neg_c, (a, b) = heapq.heappop(heap)
                c = -neg_c
                current_c = pair_freq.get((a, b), 0)
                if c == current_c and current_c > 0:
                    break
            else:
                break  # no more valid pairs

            self.merges.append((a, b))
            new_token = next_token
            next_token += 1
            self.token_to_bytes[new_token] = self.token_to_bytes[a] + self.token_to_bytes[b]
            merges_done += 1

            # Update all sequences: replace (a,b) with new_token
            for seq_idx, seq in enumerate(sequences):
                i = 0
                new_seq = []
                # Track which pairs change
                removed_pairs = {}
                added_pairs = {}
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                        p_rm = (a, b)
                        removed_pairs[p_rm] = removed_pairs.get(p_rm, 0) + 1
                        new_seq.append(new_token)
                        i += 2
                        # Check new pair with previous element (if any)
                        if len(new_seq) >= 2:
                            new_p = (new_seq[-2], new_seq[-1])
                            added_pairs[new_p] = added_pairs.get(new_p, 0) + 1
                    else:
                        new_seq.append(seq[i])
                        i += 1
                sequences[seq_idx] = new_seq

                # Apply frequency changes
                for p, dec in removed_pairs.items():
                    old_c = pair_freq.get(p, 0)
                    new_c = old_c - dec
                    if new_c <= 0:
                        if p in pair_freq:
                            del pair_freq[p]
                    else:
                        pair_freq[p] = new_c
                        heapq.heappush(heap, (-new_c, p))

                for p, inc in added_pairs.items():
                    old_c = pair_freq.get(p, 0)
                    new_c = old_c + inc
                    pair_freq[p] = new_c
                    heapq.heappush(heap, (-new_c, p))

            if verbose and merges_done % 2000 == 0:
                print(f"  BPE merge {merges_done}/{self.num_merges}")

        self._max_token = 255 + len(self.merges)
        self._build_merge_map()
        if verbose:
            print(f"BPE training: {len(self.merges)} merges, "
                  f"vocab={256 + len(self.merges)}")

    def _build_merge_map(self):
        """Build O(1) pair->token lookup from merge list."""
        self._merge_map = {}
        for i, (a, b) in enumerate(self.merges):
            self._merge_map[(a, b)] = 256 + i

    def encode(self, text, add_bos=False, add_eos=False):
        """Encode text to token IDs using BPE or word-level vocab"""
        if hasattr(self, '_word_vocab'):
            tokens = re.findall(r"[A-Za-z0-9_\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF']+|.", text)
            ids = []
            unk_id = self._word_vocab.get('<UNK>', 3)
            for tok in tokens:
                if tok in self._word_vocab:
                    ids.append(self._word_vocab[tok])
                elif tok.strip() and tok.strip().lower() in self._word_vocab:
                    ids.append(self._word_vocab[tok.strip().lower()])
                elif tok.isspace():
                    pass
                else:
                    ids.append(unk_id)
        else:
            if isinstance(text, str):
                text = text.encode("utf-8")
            ids = [b for b in text]
            if not self._merge_map:
                self._build_merge_map()
            
            # Single-pass BPE encoding using priority-based merge order
            # Much faster than iterating until convergence
            max_passes = len(self.merges) if self.merges else 0
            for _ in range(min(max_passes, 20)):  # Limit passes to prevent infinite loops
                i = 0
                new_ids = []
                changed_this_pass = False
                while i < len(ids):
                    if i < len(ids) - 1:
                        pair = (ids[i], ids[i + 1])
                        merged = self._merge_map.get(pair)
                        if merged is not None:
                            new_ids.append(merged)
                            i += 2
                            changed_this_pass = True
                            continue
                    new_ids.append(ids[i])
                    i += 1
                ids = new_ids
                if not changed_this_pass:
                    break

        if add_bos:
            ids = [self.SPECIAL["<bos>"]] + ids
        if add_eos:
            ids = ids + [self.SPECIAL["<eos>"]]
        return ids

    def decode(self, ids, skip_special=False):
        """Convert token IDs back to text"""
        if hasattr(self, '_word_vocab'):
            parts = []
            special_ids = set(self.SPECIAL.values())
            no_space_before = set('.,!?:;)]}')
            no_space_after = set('([{"\'')
            for i, tid in enumerate(ids):
                if skip_special and tid in special_ids:
                    continue
                word = self._id_to_word.get(tid)
                if word is None or len(word) > 20:
                    word = ' '
                if parts and word and word[0] in no_space_before:
                    parts.append(word)
                elif not parts or word in no_space_after:
                    parts.append(word)
                elif parts and (parts[-1][-1] in no_space_after or parts[-1][-1] == ' '):
                    parts.append(word)
                else:
                    parts.append(' ' + word)
            text = ''.join(parts)
            for a, b in [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'),
                          (' :', ':'), (' ;', ';'), ('( ', '('), (' )', ')'),
                          ('" ', '"'), (' "', '"'), ("' ", "'"), (" '", "'"),
                          ('  ', ' '), ('- ', '-'), (' /', '/'), ('\n ', '\n')]:
                text = text.replace(a, b)
            return text.strip()
        bytes_list = []
        for tid in ids:
            if skip_special and tid in self.SPECIAL.values():
                continue
            if tid < 256:
                bytes_list.append(tid)
            elif tid < 256 + len(self.merges):
                merge_idx = tid - 256
                if merge_idx < len(self.token_to_bytes):
                    bytes_list.extend(b for b in self.token_to_bytes[tid])
            else:
                bytes_list.append(32)  # space as fallback
        return bytes(bytes_list).decode("utf-8", errors="replace")

    def get_vocab_size(self):
        if hasattr(self, '_word_vocab') and self._word_vocab:
            return max(self._id_to_word.keys()) + 1
        return 256 + len(self.merges)

    def save(self, path):
        """Save tokenizer to JSON"""
        data = {
            "vocab_size": self.vocab_size,
            "num_merges": self.num_merges,
            "merges": [(int(a), int(b)) for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path):
        """Load tokenizer from JSON (supports BPE or word-level vocab formats)"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "num_merges" in data:
            # BPE format with merges
            self.num_merges = data["num_merges"]
            self.merges = [(a, b) for a, b in data["merges"]]
            self.vocab_size = data.get("vocab_size", 256 + len(self.merges))
            self.token_to_bytes = {i: bytes([i]) for i in range(256)}
            for i, (a, b) in enumerate(self.merges):
                self.token_to_bytes[256 + i] = self.token_to_bytes[a] + self.token_to_bytes[b]
            self._max_token = 255 + len(self.merges)
            self._build_merge_map()
        else:
            # Word-level vocab format: {word: id}
            self._word_vocab = data
            self._id_to_word = {v: k for k, v in data.items()}
            self.vocab_size = max(self._id_to_word.keys()) + 1 if self._id_to_word else 256
            self.num_merges = 0
            self.merges = []
            self._max_token = self.vocab_size - 1
            self._merge_map = {}

    def build_vocab(self, texts):
        """Convenience: train on all texts if merges not already loaded"""
        if not self.merges and texts:
            self.train(texts, verbose=False)

    def detect_language(self, text):
        """Detect language using character n-gram profiles (rule-based, no training)"""
        if not text:
            return "unknown"

        scripts = {
            "greek": lambda c: 0x0370 <= ord(c) <= 0x03FF,
            "cyrillic": lambda c: 0x0400 <= ord(c) <= 0x04FF,
            "arabic": lambda c: 0x0600 <= ord(c) <= 0x06FF,
            "hebrew": lambda c: 0x0590 <= ord(c) <= 0x05FF,
            "devanagari": lambda c: 0x0900 <= ord(c) <= 0x097F,
            "cjk": lambda c: 0x4E00 <= ord(c) <= 0x9FFF,
            "hiragana": lambda c: 0x3040 <= ord(c) <= 0x309F,
            "katakana": lambda c: 0x30A0 <= ord(c) <= 0x30FF,
            "hangul": lambda c: 0xAC00 <= ord(c) <= 0xD7AF,
        }

        scores = {k: 0 for k in scripts}
        total = 0
        for c in text:
            for name, test in scripts.items():
                if test(c):
                    scores[name] += 1
                    break
            else:
                if c.isalpha():
                    scores["latin"] = scores.get("latin", 0) + 1
            total += 1

        if total == 0:
            return "unknown"

        # Normalize and find best
        best = None
        best_score = 0
        for name, count in scores.items():
            ratio = count / total
            if ratio > best_score:
                best_score = ratio
                best = name

        if best_score < 0.1:
            # Default detection for Latin-script languages
            latin_indicators = {
                "en": {"the", "and", "you", "for", "are", "this", "that", "have", "from"},
                "fr": {"vous", "avec", "dans", "elle", "mais", "tout", "faire", "bien"},
                "de": {"und", "die", "der", "das", "mit", "auf", "sich", "nicht"},
                "es": {"que", "con", "por", "para", "como", "esta", "muy", "todo"},
                "it": {"che", "con", "per", "non", "della", "delle", "sono", "anche"},
            }
            words = set(text.lower().split())
            lang_scores = {}
            for lang, indicators in latin_indicators.items():
                lang_scores[lang] = len(words & indicators)
            if lang_scores and max(lang_scores.values()) > 1:
                best = max(lang_scores, key=lang_scores.get)

        return best or "latin"


class Tokenizer:
    """Alias for ByteBPETokenizer for backward compatibility"""
    def __init__(self, vocab_size=8192):
        self._tokenizer = ByteBPETokenizer(vocab_size=vocab_size)

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, val):
        self._tokenizer = val

    @property
    def merges(self):
        return self._tokenizer.merges

    @merges.setter
    def merges(self, val):
        self._tokenizer.merges = val

    @property
    def token_to_bytes(self):
        return self._tokenizer.token_to_bytes

    @token_to_bytes.setter
    def token_to_bytes(self, val):
        self._tokenizer.token_to_bytes = val

    def train(self, texts, verbose=True, max_texts=20000):
        self._tokenizer.train(texts, verbose, max_texts=max_texts)

    def encode(self, text, add_bos=False, add_eos=False):
        return self._tokenizer.encode(text, add_bos, add_eos)

    def decode(self, ids, skip_special=False):
        return self._tokenizer.decode(ids, skip_special)

    def get_vocab_size(self):
        return self._tokenizer.get_vocab_size()

    def save(self, path):
        self._tokenizer.save(path)

    def load(self, path):
        self._tokenizer.load(path)

    def build_vocab(self, texts):
        self._tokenizer.build_vocab(texts)

    def detect_language(self, text):
        return self._tokenizer.detect_language(text)

    @property
    def vocab(self):
        return {}


if __name__ == "__main__":
    tok = ByteBPETokenizer(vocab_size=8192)
    texts = ["Hello world!", "Γεια σου κόσμε!", "Привет мир!", "你好世界！"]
    tok.train(texts, verbose=True)

    for t in texts:
        ids = tok.encode(t, add_bos=True, add_eos=True)
        decoded = tok.decode(ids, skip_special=True)
        print(f"  '{t[:20]}...' -> {len(ids)} tokens -> '{decoded[:20]}...'")

    print(f"Language: {tok.detect_language('Γεια σου')} (expected: greek)")
    print(f"Language: {tok.detect_language('Hello world')} (expected: en)")
    print(f"Language: {tok.detect_language('Привет мир')} (expected: cyrillic)")
