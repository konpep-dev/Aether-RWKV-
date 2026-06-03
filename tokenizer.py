"""
ΑΠΛΟΣ TOKENIZER — μετατρέπει κείμενο σε αριθμούς (token IDs) και το αντίστροφο
Λειτουργεί σε επίπεδο λέξεων + σημείων στίξης
"""

import re
import json
from collections import Counter

class Tokenizer:
    """
    Tokenizer: μετατρέπει текст ↔ λίστα από ακέραιους αριθμούς (token IDs)
    
    Πώς λειτουργεί:
    1. Παίρνει ένα κείμενο και το σπάει σε λέξεις (tokens)
    2. Κάθε λέξη αντιστοιχεί σε ένα μοναδικό νούμερο (ID)
    3. Χτίζει το λεξιλόγιο (vocabulary) από τα δεδομένα εκπαίδευσης
    """
    
    # Ειδικά tokens (λέξεις-κλειδιά για ειδικές λειτουργίες)
    PAD = "<PAD>"   # token συμπλήρωσης (για ομοιόμορφο μήκος)
    BOS = "<BOS>"   # αρχή πρότασης (beginning of sequence)
    EOS = "<EOS>"   # τέλος πρότασης (end of sequence)  
    UNK = "<UNK>"   # άγνωστη λέξη (unknown)
    SEP = "<SEP>"   # διαχωριστής (separator)
    SPECIAL = [PAD, BOS, EOS, UNK, SEP]
    
    def __init__(self, vocab=None, max_vocab=8192):
        """
        Δημιουργεί tokenizer, είτε από υπάρχον vocab είτε από την αρχή
        
        Παράμετροι:
            vocab: υπάρχον λεξιλόγιο (dict λέξη→ID) ή None για νέο
            max_vocab: μέγιστος αριθμός λέξεων στο λεξιλόγιο
        """
        self.max_vocab = max_vocab
        if vocab:
            self.vocab = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        else:
            # Ξεκινάμε με τα ειδικά tokens
            self.vocab = {}
            self.id_to_token = {}
            for i, token in enumerate(self.SPECIAL):
                self.vocab[token] = i
                self.id_to_token[i] = token
    
    def _split(self, text):
        """
        Σπάει ένα κείμενο σε tokens (λέξεις + σημεία στίξης)
        
        Π.χ. "Hello, world!" → ["Hello", ",", "world", "!"]
        
        Χρησιμοποιεί κανονική έκφραση για να ξεχωρίσει:
        - Λέξεις (γράμματα)
        - Αριθμούς
        - Σημεία στίξης
        """
        # Βρίσκει λέξεις (γράμματα), αριθμούς, σημεία στίξης (αγνοεί κενά)
        return re.findall(r"[A-Za-z']+|[0-9]+|[.,!?;:()\[\]{}\"\-]+", text)
    
    def build_vocab(self, texts):
        """
        Χτίζει το λεξιλόγιο από λίστα κειμένων
        
        Βήματα:
        1. Σπάει όλα τα κείμενα σε λέξεις
        2. Μετράει συχνότητα κάθε λέξης
        3. Κρατάει τις πιο συχνές λέξεις (μέχρι max_vocab)
        4. Τις αντιστοιχίζει σε μοναδικά ID
        
        Παράμετροι:
            texts: λίστα από strings (τα δεδομένα εκπαίδευσης)
        """
        # Μετράμε όλες τις λέξεις σε όλα τα κείμενα
        counter = Counter()
        for text in texts:
            tokens = self._split(text)
            counter.update(tokens)
        
        # Παίρνουμε τις πιο συχνές λέξεις
        most_common = counter.most_common(self.max_vocab - len(self.SPECIAL))
        
        # Τις προσθέτουμε στο λεξιλόγιο
        for token, _ in most_common:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.id_to_token[idx] = token
        
        print(f"Vocabulary built: {len(self.vocab)} tokens")
        return self.vocab
    
    def encode(self, text, add_bos=True, add_eos=True):
        """
        Μετατρέπει κείμενο σε λίστα από token IDs
        
        Βήματα:
        1. Σπάει το κείμενο σε λέξεις
        2. Κάθε λέξη → ID (αν υπάρχει) ή UNK αν είναι άγνωστη
        3. Προσθέτει BOS/EOS αν χρειάζεται
        
        Πα настоящ:
            text: το κείμενο προς μετατροπή
            add_bos: βάζει <BOS> στην αρχή;
            add_eos: βάζει <EOS> στο τέλος;
        
        Επιστρέφει: λίστα από ακέραιους IDs
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
        Μετατρέπει λίστα από token IDs σε κείμενο
        
        Βήματα:
        1. Κάθε ID → λέξη
        2. Ενώνει τις λέξεις σε κείμενο
        
        Παράμετροι:
            ids: λίστα από ακέραιους IDs
            skip_special: αν True, αγνοεί ειδικά tokens (PAD, BOS, EOS)
        
        Επιστρέφει: string
        """
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, self.UNK)
            if skip_special and token in self.SPECIAL:
                continue
            tokens.append(token)
        
        # Ενώνει με κενό, αλλά προσέχει σημεία στίξης
        if not tokens:
            return ""
        
        text = tokens[0]
        for i in range(1, len(tokens)):
            cur = tokens[i]
            # Σημεία στίξης που ΔΕΝ θέλουν κενό πριν
            if cur in ".,!?;:)]}":
                text += cur
            else:
                text += " " + cur
        return text
    
    def get_vocab_size(self):
        """Επιστρέφει το μέγεθος του λεξιλογίου"""
        return len(self.vocab)
    
    def save(self, path):
        """Αποθηκεύει το λεξιλόγιο σε αρχείο JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    
    def load(self, path):
        """Φορτώνει το λεξιλόγιο από αρχείο JSON"""
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}


if __name__ == "__main__":
    # Test: δημιουργούμε tokenizer και δοκιμάζουμε
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
