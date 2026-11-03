import os
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
import unsloth  # noqa: F401  (must be imported before trl so it can patch it)
from trl import GRPOConfig, GRPOTrainer
import argparse
from datetime import datetime
from typing import Dict, List
import torch
from datasets import Dataset
from PIL import Image as PILImage
from unsloth import FastLanguageModel, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AskR_Solver import AskRSolver
from utils.build_prompt import build_all_subq_generation_prompt
from utils.load_dataset_utils import load_dataset_by_name
from utils.rewards import intrinsic_reward, training_reward

GLOBAL_SAVE_PATH = './outputs/train/'
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class AskRGRPOTrainer(AskRSolver):
    """GRPO training of the asker (base model + LoRA).

    GRPO samples sub-question chains from the vLLM engine with the LoRA; the reward function reuses the
    same engine without the LoRA as the frozen reasoner that answers and self-verifies them.
    """

    def __init__(self,
                 model_name: str,
                 max_tokens: int,
                 temperature: float,
                 n_verify: int,
                 max_subqs_len: int,
                 dataset_type: str,
                 gpu_memory_utilization: float,
                 max_model_len: int,
                 grpo_config: GRPOConfig,
                 lora_r: int,
                 lora_alpha: int,
                 answer_n_samples: int = 1,
                 reward_type: str = "combined",
                 visual_mode: bool = False,
                 debug: bool = False):
        super().__init__(model_name,
                         max_tokens=max_tokens,
                         temperature=temperature,
                         n_samples=1,
                         n_verify=n_verify,
                         do_verification=(reward_type == "combined"),
                         dataset_type=dataset_type,
                         gpu_memory_utilization=gpu_memory_utilization,
                         max_model_len=max_model_len,
                         max_subqs_len=max_subqs_len,
                         mode="askr",
                         skip_vllm_init=True,
                         visual_mode=visual_mode)

        self.grpo_config = grpo_config
        self.debug = debug
        self.answer_n_samples = answer_n_samples
        self.reward_type = reward_type

        # Some backbones (e.g. Gemma-3) are unstable in fp16; set ASKR_DTYPE=bf16 for them.
        train_dtype = torch.bfloat16 if os.environ.get("ASKR_DTYPE", "float16").lower() in ("bf16", "bfloat16") else torch.float16

        fast_model = FastVisionModel if visual_mode else FastLanguageModel
        model, tokenizer = fast_model.from_pretrained(
            model_name=model_name,
            max_seq_length=max_model_len,
            load_in_4bit=False,
            fast_inference=True,
            max_lora_rank=lora_r,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=train_dtype,
            unsloth_vllm_standby=os.environ["UNSLOTH_VLLM_STANDBY"] == "1",
        )
        self.tokenizer = tokenizer

        if not visual_mode:
            self.training_model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                target_modules=LORA_TARGET_MODULES,
                lora_alpha=lora_alpha,
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
        else:
            self.training_model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0,
                bias="none",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

    def _reward_askr_in_loop(
        self,
        prompts: List[str],
        completions: List,
        ground_truth: List[Dict],
        **kwargs,
    ) -> List[float]:
        """Eq. 5 reward for each sub-question chain: the reasoner answers and self-verifies it, and the
        mean verification rate is signed by final-answer correctness."""
        responses = []
        for output in completions:
            if isinstance(output, list) and len(output) > 0 and 'content' in output[0]:
                responses.append(output[0]['content'])
            else:
                responses.append(output)

        torch.cuda.empty_cache()
        if hasattr(self.model, 'wake_up'):
            self.model.wake_up()

        batch = {"questions": [], "ground_truths": [], "options_list": [], "types": [], "image_paths": []}
        input_responses = []
        for idx, gt in enumerate(ground_truth):
            for _ in range(self.answer_n_samples):
                batch["questions"].append(gt['original_prompt'])
                batch["ground_truths"].append(gt['answer'])
                batch["options_list"].append(gt.get('options'))
                batch["types"].append(gt.get('type'))
                batch["image_paths"].append(gt.get('image_paths'))
                input_responses.append(responses[idx])

        try:
            all_results = self.solve_with_askr(
                batch_samples=batch,
                askr_completions=input_responses,
                verbose=self.debug,
            )
        finally:
            if hasattr(self.model, 'sleep'):
                self.model.sleep(level=1)
            torch.cuda.empty_cache()

        rewards = []
        for idx in range(0, len(all_results), self.answer_n_samples):
            reward = 0.0
            for i in range(self.answer_n_samples):
                result = all_results[idx + i][0]
                value = result.get("is_correct") or 0.0
                if self.reward_type == "combined":
                    value = training_reward(value, intrinsic_reward(result))
                reward += value
            rewards.append(reward / self.answer_n_samples)

        return rewards

    def get_grpo_trainer(self, train_dataset) -> GRPOTrainer:
        grpo_trainer = GRPOTrainer(
            model=self.training_model,
            processing_class=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.training_model, self.tokenizer) if self.visual_mode else None,
            args=self.grpo_config,
            reward_funcs=[self._reward_askr_in_loop],
            train_dataset=train_dataset,
        )
        if getattr(grpo_trainer, 'llm', None) is None:
            raise RuntimeError("GRPOTrainer has no colocated vLLM engine; the reward function needs it")
        self.model = grpo_trainer.llm
        return grpo_trainer


def _load_image(p, max_pixels: int = 1003520) -> PILImage.Image:
    """Load an image as RGB, capping the pixel count so the visual-token count fits the context."""
    img = p if isinstance(p, PILImage.Image) else PILImage.open(p)
    img = img.convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    return img


def prepare_grpo_dataset(dataset_samples: List[Dict],
                         dataset_type: str,
                         max_subqs_len: int = 10,
                         visual_data: bool = False):
    """Turn samples into GRPO rows: a sub-question generation prompt plus the ground truth the reward needs."""
    train_samples = []
    for sample in dataset_samples:
        prompt = sample.get('prompt') or sample.get('question')
        answer = sample.get('answer')
        image_paths = sample.get('image_paths', None)
        options = sample.get('options')
        qtype = sample.get('type') or dataset_type

        if prompt is None or answer is None or (visual_data and image_paths is None):
            continue

        sys_p, usr_p = build_all_subq_generation_prompt(prompt, qtype, options, max_subqs_len=max_subqs_len)
        if image_paths is not None:
            user_content = [{"type": "image"}, {"type": "text", "text": usr_p}]
        else:
            user_content = usr_p

        ground_truth = {
            "original_prompt": prompt,
            "answer": answer,
            "image_paths": image_paths,
            "type": qtype,
        }
        if options is not None:
            ground_truth["options"] = options
        new_sample = {
            'prompt': [{"role": "system", "content": sys_p}, {"role": "user", "content": user_content}],
            'ground_truth': ground_truth,
        }
        if visual_data:
            new_sample['images'] = [_load_image(p) for p in image_paths]
        train_samples.append(new_sample)

    print(f"Training samples: {len(train_samples)}")
    return train_samples if visual_data else Dataset.from_list(train_samples)


def main():
    parser = argparse.ArgumentParser(description="Train the AskR asker with GRPO")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Asker (and frozen reasoner) backbone")
    parser.add_argument("--dataset", type=str, default="math_stem_mixed", choices=["math_stem_mixed", "lpt30k", "math", "stem"])
    parser.add_argument("--dataset-samples", type=int, default=5000, help="Number of training problems (-1 = all)")
    parser.add_argument("--visual-mode", action="store_true", help="Vision-language training (use with lpt30k and a Qwen-VL model)")
    parser.add_argument("--output-dir", type=str, default="askr", help="Run name; saved under ./outputs/train/<name>_<timestamp>")

    # Reward
    parser.add_argument("--reward-type", type=str, default="combined", choices=["combined", "answer"],
                        help="'combined' = Eq. 5 (intrinsic reward signed by correctness); 'answer' = correctness only (ablation)")
    parser.add_argument("--max-subq-steps", type=int, default=10, help="Maximum number of sub-questions per chain")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.3, help="Reasoner temperature when computing rewards")
    parser.add_argument("--answer-n-samples", type=int, default=1, help="Reasoner answers averaged per sub-question chain")
    parser.add_argument("--n-verify", type=int, default=1, help="Self-verification rounds per SubQ-SubA pair")

    # GRPO
    parser.add_argument("--n-samples", type=int, default=8, help="GRPO rollouts per prompt")
    parser.add_argument("--train-temperature", type=float, default=0.8, help="Asker temperature for GRPO rollouts")
    parser.add_argument("--num-train-epochs", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)

    # vLLM
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-model-len", type=int, default=16384)

    # Logging and checkpoints
    parser.add_argument("--logging-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--use-wandb", action="store_true", help="Log to Weights & Biases (project from WANDB_PROJECT)")
    parser.add_argument("--debug", action="store_true", help="Verbose reward computation")

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.output_dir}_{timestamp}"
    output_dir = os.path.join(GLOBAL_SAVE_PATH, run_name)
    os.makedirs(output_dir, exist_ok=True)

    dataset_samples = load_dataset_by_name(args.dataset, args.dataset_samples)
    train_dataset = prepare_grpo_dataset(
        dataset_samples,
        dataset_type=args.dataset,
        max_subqs_len=args.max_subq_steps,
        visual_data=args.visual_mode)

    use_bf16 = os.environ.get("ASKR_DTYPE", "float16").lower() in ("bf16", "bfloat16")
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        beta=args.kl_beta,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.n_samples,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length=args.max_tokens,
        max_prompt_length=args.max_model_len - args.max_tokens,
        temperature=args.train_temperature,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        loss_type="dapo",
        epsilon_high=0.28,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        fp16=not use_bf16,
        bf16=use_bf16,
        report_to="wandb" if args.use_wandb else "none",
        log_level="info",
    )

    trainer_wrapper = AskRGRPOTrainer(
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n_verify=args.n_verify,
        max_subqs_len=args.max_subq_steps,
        dataset_type=args.dataset,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        grpo_config=grpo_config,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        answer_n_samples=args.answer_n_samples,
        reward_type=args.reward_type,
        visual_mode=args.visual_mode,
        debug=args.debug,
    )
    trainer = trainer_wrapper.get_grpo_trainer(train_dataset)

    print(f"\n{'='*70}\nStarting GRPO training...\n{'='*70}\n")
    trainer.train()

    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
