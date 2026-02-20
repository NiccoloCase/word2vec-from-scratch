from collections import Counter
from typing import List, Dict
import random as rnd
import numpy as np

class Vocabulary:
    def __init__(
        self,
        tokens: List[str], 
        min_count: int = 5, # the minimum frequency for a word to be included in the vocabulary
        subsample_t: float = 1e-3 # the threshold for subsampling (Mikolov)
    ) -> None:
        self.min_count = min_count
        self.subsample_t = subsample_t
        self.freqs: Dict[str, int] = {}
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

        self.counts: List[int] = []     
        self.keep_prob: np.ndarray | None = None

        self._build(tokens)
        self.set_subsampling(subsample_t)

    def _build(self, tokens: List[str]) -> None:
        counts = Counter(tokens)

        # Filter by min_count
        vocab_items = [
            (word, freq)
            for word, freq in counts.items() if freq >= self.min_count
        ]

        # Sort by frequency 
        vocab_items.sort(key=lambda x: x[1], reverse=True)
      
        index = 0  

        # add vocabulary
        for word, freq in vocab_items:
            self.word2idx[word] = index
            self.idx2word[index] = word
            self.freqs[word] = freq
            self.counts.append(freq)
            index += 1

    def encode(self, tokens: List[str]) -> List[int]:
        "Encode a list of tokens into their corresponding indices"
        tokens = [self.word2idx.get(token, -1) for token in tokens]
        return [token for token in tokens if token != -1]

    def decode(self, indices: List[int]) -> List[str]:
        "Decode a list of indices back into their corresponding tokens"
        return [self.idx2word.get(idx, -1) for idx in indices if self.idx2word.get(idx, -1) != -1]
    
    def set_subsampling(self, t: float = 1e-3) -> None:
        """
        Precompute keep probabilities using Mikolov subsampling.
        Specials are always kept 
        """
        counts = np.asarray(self.counts, dtype=np.float64)
        total = counts.sum()

        if total <= 0:
            self.keep_prob = np.ones_like(counts, dtype=np.float64)
            return

        freq = counts / total

        keep = np.ones_like(freq, dtype=np.float64)
        mask = counts > 0 # only compute for words that appear at least once
        keep[mask] = np.sqrt(t / freq[mask])
        np.clip(keep, 0.0, 1.0, out=keep)

        self.keep_prob = keep
    
    def encode_subsampled(self, tokens: list[str]) -> List[int]:
        """
        Encode tokens while applying Mikolov subsampling:
            P(keep) = min(1, sqrt(t / f(w)) where f(w) is relative frequency

        Returns a list of indices for the tokens that are kept after subsampling.
        """
        if self.keep_prob is None:
            raise RuntimeError("Subsampling not initialized. Call set_subsampling().")

        indices: List[int] = []
        for w in tokens:
            idx = self.word2idx.get(w)
            if idx is None: continue

            if rnd.random() < float(self.keep_prob[idx]): indices.append(idx)

        return indices

    def to_state(self) -> dict:
        """Return a serialisable snapshot of the vocabulary"""
        words = [self.idx2word[i] for i in range(len(self.idx2word))]
        freqs = [self.freqs[w] for w in words]
        return {
            "words": words,
            "freqs": freqs,
            "counts": list(self.counts),
            "min_count": self.min_count,
            "subsample_t": self.subsample_t,
        }

    @classmethod
    def from_state(cls, state: dict) -> "Vocabulary":
        """Rebuild a vocabulary from a snapshot created by to_state."""
        words = state.get("words", [])
        freqs = state.get("freqs", [])
        counts = state.get("counts", freqs)
        min_count = state.get("min_count", 5)
        subsample_t = state.get("subsample_t", 1e-3)

        voc = cls(tokens=[], min_count=min_count, subsample_t=subsample_t)
        voc.word2idx = {w: i for i, w in enumerate(words)}
        voc.idx2word = {i: w for i, w in enumerate(words)}
        voc.freqs = {w: int(f) for w, f in zip(words, freqs)}
        voc.counts = [int(c) for c in counts]
        voc.set_subsampling(subsample_t)
        return voc


    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, token: str) -> bool:
        return token in self.word2idx