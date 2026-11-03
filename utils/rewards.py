from typing import Dict


def intrinsic_reward(sample: Dict) -> float:
    """Eq. 4: mean self-verification verdict over a trajectory's SubQA pairs."""
    rates = [p.get("verification_rate", 0) for p in sample.get("subqa_pairs", [])]
    return sum(rates) / len(rates) if rates else 0.0


def training_reward(correctness: float, verification_rate: float) -> float:
    """Eq. 5: R = (2V - 1) * r, with V = 1 when the correctness score (0/1, or the LLM-judge score) exceeds 0.5."""
    return verification_rate if correctness > 0.5 else -verification_rate
