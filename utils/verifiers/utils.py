# This code base is based on Genesys (https://github.com/PrimeIntellect-ai/genesys)
import re
import json

def extract_json(text):
    """Extract JSON from text with multiple fallback strategies."""
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', text, re.IGNORECASE)
    score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)

    if explanation_match and score_match:
        try:
            return {
                "explanation": explanation_match.group(1),
                "score": float(score_match.group(1))
            }
        except (ValueError, AttributeError):
            pass

    score_match = re.search(r'score[:\s]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if score_match:
        return {
            "explanation": "Score extracted without full JSON",
            "score": float(score_match.group(1))
        }

    raise ValueError(f"Failed to parse JSON from the extracted content. Text preview: {text[:200]}...")