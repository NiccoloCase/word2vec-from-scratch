import re
import numpy as np

def tokenise(text: str) -> list[str]:
    "Very simple but effective tokeniser: we just split on non-alphabetic characters and lowercase everything"
    tokens = re.findall(r"[a-z]+", text.lower())
    return tokens

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


def learning_rate_decay(lr0: float, lr_min: float, current_step: int, total_steps: int) -> float:
    """
    Computes the learning rate using linear decay scheduling.
    
    The learning rate starts at lr0 and linearly decreases to lr_min over total_steps.

    Args:
        lr0: Initial learning rate.
        lr_min: Minimum learning rate (floor value).
        current_step: Current training step/iteration.
        total_steps: Total number of training steps.
    
    Returns:
        float: Current learning rate.

    """
    return max(lr_min, lr0 * (1.0 - current_step / total_steps))

