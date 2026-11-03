import os
import time
from typing import List, Optional
from tqdm import tqdm
from vllm import SamplingParams
from PIL import Image

def load_images(paths: List) -> List[Image.Image]:
    """Load images from paths or PIL Image objects.

    Args:
        paths: List where each element can be:
            - A string path: "path/to/image.jpg"
            - A list/tuple with one string path: ["path/to/image.jpg"]
            - A PIL Image object
            - A list/tuple with one PIL Image: [PIL.Image]

    Returns:
        List of PIL Image objects (RGB mode)

    Raises:
        FileNotFoundError: If any image path doesn't exist
        Exception: If any image fails to load
    """
    images = []
    for i, path in enumerate(paths):
        if isinstance(path, (list, tuple)) and len(path) > 0:
            path = path[0]

        if isinstance(path, Image.Image):
            images.append(path.convert("RGB"))
        elif isinstance(path, str):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found at index {i}: {path}")
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                raise Exception(f"Failed to load image at index {i} ({path}): {e}")
        else:
            raise ValueError(f"Invalid image format at index {i}: {type(path)}. Expected string path or PIL Image.")

    return images

def generate_text_batch(
    system_prompt: str | List[str],
    user_prompt: str | List[str],
    batch_size: int,
    images_paths: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
    tokenizer = None,
    temperature: float = 0.8,
    lora_request: Optional[dict] = None,
    model = None,
    enable_lora: bool = False,
    verbose: bool = True,
) -> List[str]:
    """Generate multiple text samples using vLLM batch inference.

    Handles both uniform batches (same prompt replicated) and heterogeneous batches
    (different prompts per sample). Supports multi-modal generation with images.

    Args:
        system_prompt: Single string or list of system prompts
        user_prompt: Single string or list of user prompts
        batch_size: Number of samples to generate
        images_paths: Optional list of image paths (length must equal batch_size)
        max_tokens: Maximum tokens to generate
        tokenizer: Tokenizer for chat template
        temperature: Sampling temperature
        lora_request: Optional LoRA request
        model: vLLM model instance
        enable_lora: Whether to enable LoRA
        verbose: Whether to show progress

    Returns:
        List of generated text responses
    """
    prompts = []
    images_list = []

    for sys_p, usr_p, image_p in zip(system_prompt, user_prompt, images_paths or [None]*batch_size):
        if image_p is not None:
            if isinstance(image_p[0], str):
                image_p = load_images([image_p])
            images_list.append(image_p[0])
            
            messages = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": usr_p},
                ]}
            ]
        else:
            images_list.append(None)
            messages = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p}
            ]
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        skip_special_tokens=True
    )

    start = time.time()
    if any(img is not None for img in images_list):
        vllm_input = [{
                "prompt": p,
                "multi_modal_data": {"image": img} if img else None
            } for p, img in zip(prompts, images_list)]
    else:
        vllm_input = prompts
    gen_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": verbose,
    }
    if lora_request is not None and enable_lora:
        gen_kwargs["lora_request"] = lora_request
    outputs = model.generate(vllm_input, **gen_kwargs)
    end = time.time()

    responses = []
    total_generated = 0
    for output in outputs:
        response_text = output.outputs[0].text.strip()
        responses.append(response_text)
        total_generated += len(output.outputs[0].token_ids)

    if verbose:
        tqdm.write(f"Throughput: {total_generated / (end - start):.2f} toks/sec")

    return responses