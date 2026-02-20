import re
import numpy as np

def tokenise(text: str) -> list[str]:
    "Very simple but effective tokeniser: we just split on non-alphabetic characters and lowercase everything"
    tookens = re.findall(r"[a-z]+", text.lower())
    return tookens


def build_noise_distribution(freqs: list[int], power: float = 0.75) -> np.ndarray:
    """
    Unigram distribution raised to the 3/4 power.
    Used to draw negative samples, preventing very frequent words from dominating.
    """
    arr = np.array(freqs, dtype=np.float64) ** power
    return arr / arr.sum()


def sigmoid(x: np.ndarray) -> np.ndarray:
    "Sigmoid function"
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )