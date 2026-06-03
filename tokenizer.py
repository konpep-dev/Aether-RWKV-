"""
SIMPLE TOKENIZER — converts text to numbers (token IDs) and back
Operates at the level of words + punctuation marks
"""

import re
import json
from collections import Counter

class Tokenizer:
    """
    Tokenizer: converts text ↔ list of integers (token IDs)
    
    How it works:
    1. Takes a text and splits it into words (tokens)
    2. Each word corresponds to a unique number (ID)
    3. Builds the vocabulary from the training data
    """
    
    # Special tokens (keywords for special functions)
    PAD = "<PAD>"   # padding token (for uniform length)
    BOS = "<BOS>"   # beginning of sequence
    EOS = "<EOS>"   # end of sequence  
    UNK = "<UNK>"   # unknown word
    SEP = "<SEP>"   # separator
    SPECIAL = [PAD, BOS, EOS, UNK, SEP]
    
    def __init__(self, vocab=None, max_vocab=8192):
        """
        Creates a tokenizer, either from an existing vocab or from scratch
        
        Parameters:
            vocab: existing vocabulary (dict word→ID) or None for a new one
            max_vocab: maximum number of words in the vocabulary
        """
        self.max_vocab = max_vocab
        if vocab:
            self.vocab = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        else:
            # Start with the special tokens
            self.vocab = {}
            self.id_to_token = {}
            for i, token in enumerate(self.SPECIAL):
                self.vocab[token] = i
                self.id_to_token[i] = token
    
    def _split(self, text):
        """
        Splits a text into tokens (words + punctuation marks)
        
        E.g. "Hello, world!" → ["Hello", ",", "world", "!"]
        
        Uses a regular expression to separate:
        - Words (letters)
        - Numbers
        - Punctuation marks
        """
        # Finds words (letters), numbers, punctuation marks (ignores whitespace)
        return re.findall(r"[A-Za-z']+|[0-9]+|[.,!?;:()\[\]{}\"\-]+", text)
    
    def build_vocab(self, texts):
        """
        Builds the vocabulary from a list of texts
        
        Steps:
        1. Splits all texts into words
        2. Counts the frequency of each word
        3. Keeps the most frequent words (up to max_vocab)
        4. Maps them to unique IDs
        
        Parameters:
            texts: list of strings (the training data)
        """
        # Count all words across all texts
        counter = Counter()
        for text in texts:
            tokens = self._split(text)
            counter.update(tokens)
        
        # Get the most frequent words
        most_common = counter.most_common(self.max_vocab - len(self.SPECIAL))
        
        # Add them to the vocabulary
        for token, _ in most_common:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.id_to_token[idx] = token
        
        print(f"Vocabulary built: {len(self.vocab)} tokens")
        return self.vocab
    
    def encode(self, text, add_bos=True, add_eos=True):
        """
        Converts text to a list of token IDs
        
        Steps:
        1. Splits the text into words
        2. Each word → ID (if it exists) or UNK if unknown
        3. Adds BOS/EOS if requested
        
        Parameters:
            text: the text to convert
            add_bos: prepend <BOS>?
            add_eos: append <EOS>?
        
        Returns: list of integer IDs
        """
        tokens = self._split(text)
        ids = []
        if add_bos:
            ids.append(self.vocab[self.BOS])
        
        unk_id = self.vocab.get(self.UNK, 0)
        for token in tokens:
            ids.append(self.vocab.get(token, unk_id))
        
        if add_eos:
            ids.append(self.vocab[self.EOS])
        
        return ids
    
    def decode(self, ids, skip_special=False):
        """
        Converts a list of token IDs to text
        
        Steps:
        1. Each ID → word
        2. Joins the words into text
        
        Parameters:
            ids: list of integer IDs
            skip_special: if True, ignores special tokens (PAD, BOS, EOS)
        
        Returns: string
        """
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, self.UNK)
            if skip_special and token in self.SPECIAL:
                continue
            tokens.append(token)
        
        # Join with spaces, but be careful with punctuation
        if not tokens:
            return ""
        
        text = tokens[0]
        for i in range(1, len(tokens)):
            cur = tokens[i]
            # Punctuation that should NOT have a space before it
            if cur in ".,!?;:)]}":
                text += cur
            else:
                text += " " + cur
        return text
    
    def get_vocab_size(self):
        """Returns the size of the vocabulary"""
        return len(self.vocab)
    
    def save(self, path):
        """Saves the vocabulary to a JSON file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    
    def load(self, path):
        """Loads the vocabulary from a JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}


if __name__ == "__main__":
    # Test: create a tokenizer and try it out
    sample_texts = [
        "Hello, world! I am Aether.",
        "User: Who are you?\n\nAether: I am Aether.",
    ]
    
    tok = Tokenizer()
    tok.build_vocab(sample_texts)
    
    encoded = tok.encode("Hello, Aether!")
    decoded = tok.decode(encoded, skip_special=True)
    
    print(f"Vocab size: {tok.get_vocab_size()}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
