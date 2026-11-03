from typing import Dict
import json
import gzip

FORMATS = ("json", "json.gz", "jsonl", "jsonl.gz")


def save_results(data: Dict, output_file: str, output_format: str = "json") -> str:
    """Save results as json / jsonl, optionally gzipped. Returns the written path.

    jsonl writes one line per question, then the summary (without results) marked "_summary": true.
    """
    if output_format not in FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}")

    base_path = output_file
    for ext in sorted(FORMATS, key=len, reverse=True):
        if base_path.endswith("." + ext):
            base_path = base_path[:-len(ext) - 1]
            break
    final_path = f"{base_path}.{output_format}"

    opener = gzip.open if output_format.endswith(".gz") else open
    with opener(final_path, "wt", encoding="utf-8") as f:
        if output_format.startswith("jsonl"):
            for result in data.get("results", []):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            summary = {k: v for k, v in data.items() if k != "results"}
            summary["_summary"] = True
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return final_path
