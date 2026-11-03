import re
from typing import Tuple
from math_verify import verify
from latex2sympy2_extended import latex2sympy

MULTIPLE_CHOICE_TYPES = ["mmlu", "lab-bench", "scieval", "vstar", "cvbench", "mmstar", "lpt30k"]

_OPTION_RE = re.compile(r'(?:^|\n|\s)([A-J])\.\s+(.+?)(?=(?:\s+[A-J]\.\s)|\s*Answer:\s*$|$)', re.S)


def _normalize_option_text(text: str) -> str:
    return re.sub(r'[\s,$\\*_`]|\\text\{|\}|\\,', '', str(text)).strip().lower().rstrip('.')


def _option_letter_for(predicted_answer: str, prompt: str):
    """Map an answer written as option content ("C. 0.023 N", "**A**", "3.00M") to its letter. None if ambiguous."""
    pred = re.sub(r'[*_`]', '', str(predicted_answer)).strip()

    m = re.match(r'^\(?([A-J])\)?[\.\):]?\s*$', pred, re.I)
    if m:
        return m.group(1).upper()
    m = re.match(r'^\(?([A-J])\)?[\.\):]\s+\S', pred, re.I)
    if m:
        return m.group(1).upper()

    body = prompt.rsplit('Answer:', 1)[0] if prompt else ''
    options = {l: t.strip() for l, t in _OPTION_RE.findall(body)}
    if not options:
        return None
    norm_pred = _normalize_option_text(pred)
    if not norm_pred:
        return None

    exact = [l for l, t in options.items() if _normalize_option_text(t) == norm_pred]
    if len(exact) == 1:
        return exact[0]
    partial = [l for l, t in options.items()
               if norm_pred in _normalize_option_text(t) or _normalize_option_text(t) in norm_pred]
    if len(partial) == 1:
        return partial[0]
    return None


def verify_answer(predicted_answer: str, ground_truth: str, prompt: str = None, dataset_type: str = None) -> Tuple[float, str]:
    """Rule-based check for multiple-choice and math answers. STEM answers are scored by LlmJudgeVerifier."""
    if not predicted_answer or not ground_truth:
        return 0.0, "Missing answer or ground truth"

    if dataset_type in MULTIPLE_CHOICE_TYPES:
        pred_normalized = predicted_answer.strip().upper()
        gt_normalized = ground_truth.strip().upper()
        is_correct = pred_normalized == gt_normalized
        if not is_correct and prompt and re.fullmatch(r'[A-J]', gt_normalized):
            letter = _option_letter_for(predicted_answer, prompt)
            if letter is not None:
                return (float(letter == gt_normalized),
                        f"{dataset_type.upper()} option-content match "
                        f"(pred -> {letter}, GT: {gt_normalized})")
        return float(is_correct), f"{dataset_type.upper()} exact match (GT: {gt_normalized})"

    if dataset_type == "math":
        try:
            gold_parsed = latex2sympy(ground_truth)
            answer_parsed = latex2sympy(predicted_answer)
            is_correct = verify(gold_parsed, answer_parsed, timeout_seconds=15)
            return float(is_correct), f"Verified successfully (GT: {ground_truth})"
        except Exception as e:
            pred_normalized = predicted_answer.strip().lower().replace(" ", "").replace("\\", "")
            gt_normalized = ground_truth.strip().lower().replace(" ", "").replace("\\", "")
            is_correct = pred_normalized == gt_normalized
            return float(is_correct), f"Parse error: {str(e)}, used string matching (GT: {ground_truth})"

    raise ValueError(f"verify_answer does not handle dataset_type={dataset_type!r}")
