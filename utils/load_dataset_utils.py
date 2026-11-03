"""
Dataset loaders. Each returns a list of samples with at least:
- 'prompt': the question (str)
- 'answer': the ground truth (str)
and optionally 'options', 'image_paths', 'type' (per-sample dataset type) and 'metadata'.
"""
import os
import re
import ast
import json
import random
import hashlib
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LPT_DATA_DIR = os.path.join(DATA_DIR, "LongPerceptualThoughts-30k")
VSTAR_IMAGE_DIR = os.path.join(DATA_DIR, "vstar_images")


def _stable_shuffle(items, key: str):
    """Shuffle `items` in place with a permutation seeded by `key`, so option order is identical across runs."""
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(items)


def load_dataset_by_name(dataset_name: str, num_samples: int = 10):
    loaders = {
        "math": load_math_dataset,
        "stem": load_stem_dataset,
        "math_stem_mixed": load_math_stem_mixed_dataset,
        "lpt30k": load_lpt30k_dataset,
        "mmlu": load_mmlu_dataset,
        "lab-bench": load_labbench_dataset,
        "scieval": load_scieval_dataset,
        "vstar": load_vstar_dataset,
        "cvbench": load_cvbench_dataset,
        "mmstar": load_mmstar_dataset,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return loaders[dataset_name](num_samples)


def extract_ground_truth(verification_info):
    """Ground truth from SYNTHETIC-1's `verification_info` (a dict or its string repr)."""
    if isinstance(verification_info, str):
        verification_info = ast.literal_eval(verification_info)
    if isinstance(verification_info, dict):
        return verification_info.get('ground_truth', '')
    return ''


def load_math_dataset(num_samples: int = 10):
    """SYNTHETIC-1 verifiable math problems."""
    dataset_name = "PrimeIntellect/verifiable-math-problems"
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    tqdm.write(f"Dataset loaded: {len(dataset)} samples\n")
    if num_samples != -1:
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    return [{
        'prompt': sample.get('prompt', ''),
        'answer': extract_ground_truth(sample['verification_info']),
        'type': 'math',
    } for sample in dataset]


def load_stem_dataset(num_samples: int = 10):
    """SYNTHETIC-1 StackExchange STEM questions (graded by the LLM judge)."""
    dataset_name = "PrimeIntellect/stackexchange-question-answering"
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    tqdm.write(f"Dataset loaded: {len(dataset)} samples\n")
    if num_samples != -1:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return [{
        'prompt': sample.get('prompt', ''),
        'answer': sample.get('answer', sample.get('gold_standard_solution', None)),
        'type': 'stem',
    } for sample in dataset]


def load_math_stem_mixed_dataset(num_samples: int = 10):
    """The 5K math + STEM training set."""
    half_num_samples = num_samples // 2 if num_samples != -1 else -1
    print(f"Loading mixed dataset: {MIXED_DATA_PATH}...")
    math_data = load_math_dataset(num_samples=half_num_samples)
    stem_data = load_stem_dataset(num_samples=half_num_samples)
    mixed_data = math_data + stem_data
    random.shuffle(mixed_data)

    print(f"Mixed dataset size: {len(mixed_data)} samples\n")
    return mixed_data


def load_lpt30k_dataset(num_samples: int = 10):
    """The LongPerceptualThoughts-30K questions"""
    print("Loading dataset: LongPerceptualThoughts/LPT-30K...")
    with open(os.path.join(LPT_DATA_DIR, "LongPerceptualThoughts_SFT.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Dataset loaded: {len(data)} samples\n")

    dataset_samples = []
    for idx, item in enumerate(data):

        question = ""
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                question = msg.get("content", "").replace("<image>", "").strip()
                break

        gt = ""
        for msg in item.get("messages", []):
            if msg.get("role") == "assistant":
                match = re.search(r'<answer>\s*\(?([A-D])\)?\s*</answer>', msg.get("content", ""), re.I)
                if match:
                    gt = match.group(1).upper()
                break

        if not question or not gt:
            continue

        image_paths = [os.path.join(LPT_DATA_DIR, "docci/images", os.path.basename(img)) for img in item.get("images", [])]
        dataset_samples.append({
            "prompt": question,
            "answer": gt,
            "image_paths": [image_paths[0]],
        })

    if num_samples != -1:
        dataset_samples = dataset_samples[:num_samples]

    print(f"Loaded {len(dataset_samples)} LPT-30K samples\n")
    return dataset_samples


def load_mmlu_dataset(num_samples: int = 10):
    """MMLU-Pro test split, restricted to seven STEM categories."""
    dataset_name = "TIGER-Lab/MMLU-Pro"
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="test")
    tqdm.write(f"Dataset loaded: {len(dataset)} samples\n")
    subjects_cnt = {
        'physics': 0,
        'chemistry': 0,
        'computer science': 0,
        'math': 0,
        'biology': 0,
        'health': 0,
        'engineering': 0
    }
    max_per_subject = num_samples // len(subjects_cnt) if num_samples != -1 else -1

    dataset_samples = []
    for sample in dataset:
        subject = sample.get('category', '').lower()
        if subject in subjects_cnt and (max_per_subject == -1 or subjects_cnt[subject] < max_per_subject):
            subjects_cnt[subject] += 1
            dataset_samples.append({
                'prompt': sample.get('question', ''),
                'options': sample.get('options', []),
                'answer': sample.get('answer', None),
                'metadata': {'subject': subject},
            })
    tqdm.write(f"Loaded {len(dataset_samples)} MMLU samples across {len([s for s in subjects_cnt.values() if s > 0])} subjects")
    tqdm.write(f"Subject distribution: {subjects_cnt}\n")
    return dataset_samples


def load_labbench_dataset(num_samples: int = 10):
    """LAB-Bench CloningScenarios + ProtocolQA + SeqQA; options are the distractors plus the ideal answer."""
    dataset_name = "futurehouse/lab-bench"
    print(f"Loading dataset: {dataset_name}...")
    dataset = concatenate_datasets([
        load_dataset(dataset_name, subset, split="train")
        for subset in ('CloningScenarios', 'ProtocolQA', 'SeqQA')
    ])
    tqdm.write(f"Dataset loaded: {len(dataset)} samples\n")

    if num_samples != -1:
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    dataset_samples = []
    for sample in dataset:
        question = sample.get('question', '')
        answer = sample.get('ideal', '')
        options = list(sample.get('distractors', [])) + [answer]
        _stable_shuffle(options, question)
        dataset_samples.append({
            'prompt': question,
            'options': options,
            'answer': chr(65 + options.index(answer)),
        })

    tqdm.write(f"Loaded {len(dataset_samples)} Lab-Bench samples\n")
    return dataset_samples


def load_scieval_dataset(num_samples: int = 10):
    """SciEval test split, questions with a single gold answer."""
    dataset_name = "OpenDFM/SciEval"
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="test")
    dataset_samples = []
    category_stats = {}
    for sample in dataset:
        answer = sample.get('answer', [])
        category = sample.get('category', '')
        category_stats[category] = category_stats.get(category, 0) + 1
        if not answer or len(answer) != 1:
            continue
        dataset_samples.append({
            'prompt': sample.get('question', ''),
            'answer': answer[0],
            'metadata': {'category': category},
        })
    if num_samples != -1:
        random.seed(42)
        dataset_samples = random.sample(dataset_samples, min(num_samples, len(dataset_samples)))
    tqdm.write(f"Dataset loaded: {len(dataset_samples)} samples\n")
    print("Category distribution in loaded samples:")
    for category, count in category_stats.items():
        print(f"  {category}: {count}")
    return dataset_samples


def load_vstar_dataset(num_samples: int = 10):
    """V* Bench; images are downloaded from the HF repo into data/vstar_images."""
    from huggingface_hub import hf_hub_download

    dataset = load_dataset('craigwu/vstar_bench', split="test")
    tqdm.write(f"Dataset loaded: {len(dataset)} samples\n")
    cache_dir = VSTAR_IMAGE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    dataset_samples = []
    for sample in dataset:
        image_path_in_repo = sample.get('image', '')
        if not image_path_in_repo:
            continue
        try:
            local_image_path = hf_hub_download(
                repo_id='craigwu/vstar_bench',
                filename=image_path_in_repo,
                repo_type='dataset',
                cache_dir=cache_dir
            )
        except Exception as e:
            tqdm.write(f"Failed to download image {image_path_in_repo}: {e}")
            continue

        # Text format: "Question\n(A) opt1\n(B) opt2\n...\nAnswer with..."
        text = sample.get('text', '')
        question_parts = []
        options = []
        for line in text.split('\n'):
            option_match = re.match(r'\(([A-D])\)\s*(.+)', line.strip())
            if option_match:
                options.append(option_match.group(2).strip())
            elif not line.strip().startswith('Answer with'):
                question_parts.append(line)
        question = '\n'.join(question_parts).strip()
        if not options:
            question = text
            options = None

        dataset_samples.append({
            'prompt': question,
            'answer': sample.get('label', ''),
            'image_paths': [local_image_path],
            'options': options,
            'metadata': {'category': sample.get('category', '')},
        })
        if num_samples != -1 and len(dataset_samples) >= num_samples:
            break

    tqdm.write(f"Loaded {len(dataset_samples)} VStar samples with images\n")
    return dataset_samples


def load_cvbench_dataset(num_samples: int = 10):
    """CV-Bench 2D test split."""
    dataset = load_dataset('nyu-visionx/CV-Bench', '2D', split="test")
    tqdm.write(f"Dataset loaded: {len(dataset)} samples\n")

    dataset_samples = []
    for sample in dataset:
        dataset_samples.append({
            'prompt': sample.get('question', ''),
            'answer': re.sub(r'[\(\)]', '', sample.get('answer', '')).strip(),
            'image_paths': [sample.get('image', '')],
            'options': sample.get('choices', None),
        })
        if num_samples != -1 and len(dataset_samples) >= num_samples:
            break

    tqdm.write(f"Loaded {len(dataset_samples)} CVBench samples with images\n")
    return dataset_samples


def load_mmstar_dataset(num_samples: int = 10):
    """MMStar val split, perception and instance-reasoning categories."""
    dataset = load_dataset('Lin-Chen/MMStar', split="val")
    tqdm.write(f"Dataset loaded: {len(dataset)} samples\n")
    selected_categories = ['coarse perception', 'fine-grained perception', 'instance reasoning']

    dataset_samples = []
    for sample in dataset:
        if sample.get('category', '') not in selected_categories:
            continue
        dataset_samples.append({
            'prompt': sample.get('question', ''),
            'answer': sample.get('answer', ''),
            'image_paths': [sample.get('image', '')],
            'options': sample.get('choices', None),
            'metadata': {'category': sample.get('category', '')},
        })
        if num_samples != -1 and len(dataset_samples) >= num_samples:
            break

    tqdm.write(f"Loaded {len(dataset_samples)} MMStar samples with images\n")
    return dataset_samples
