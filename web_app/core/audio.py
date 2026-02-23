
import numpy as np

def _rms(audio: np.ndarray) -> float:
    """Return the root-mean-square energy of a float32 PCM buffer."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))
