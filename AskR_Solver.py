#!/usr/bin/env python3
import argparse
import time
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm.lora.request import LoRARequest
from tqdm import tqdm
from utils.verifiers.llm_judge_verifier import LlmJudgeVerifier
from utils.build_prompt import *
from utils.extract_parse import *
from utils.load_dataset_utils import *
from utils.tmpc_utils import *
from utils.rewards import intrinsic_reward
from utils.save_utils import save_results
from utils.generation_utils import generate_text_batch
from utils.verifiers.general_verifier import verify_answer

EVAL_DATASETS = ["mmlu", "lab-bench", "scieval", "vstar", "mmstar", "cvbench"]


class AskRSolver:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 4096,
        temperature: float = 0.8,
        n_samples: int = 8,
        n_verify: int = 1,
        do_verification: bool = True,
        dataset_type: str = "math",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 16384,
        max_subqs_len: Optional[int] = 10,
        skip_vllm_init: bool = False,
        lora_path: Optional[str] = None,
        visual_mode: bool = False,
        no_think: bool = False,
        mode: str = "askr",
        asker_model_name: Optional[str] = None,
        asker_gpu_memory_utilization: float = 0.20,
        asker_max_model_len: Optional[int] = None,
        asker_tensor_parallel_size: int = 1,
        n_iterations: int = 3,
        n_mpc_samples: int = 8,
        n_final_samples: int = 8,
        threshold: float = 0.5,
        max_per_segment: int = 3,
        max_final_retries: int = 1,
        save_buffer_details: bool = True,
    ):
        """
        mode: "askr" samples n_samples SubQA trajectories per question; "tmpc" adds test-time optimization.
        asker_model_name: if set, a second vLLM engine (with the LoRA) generates sub-questions and
            `model_name` only answers and verifies them (the reasoner-scale setting).
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n_samples = n_samples
        self.n_verify = n_verify
        self.do_verification = True if mode == "tmpc" else do_verification
        self.dataset_type = dataset_type
        self.max_model_len = max_model_len
        self.max_subqs_len = max_subqs_len
        self.lora_path = lora_path
        self.visual_mode = visual_mode
        self.mode = mode

        self.n_iterations = n_iterations
        self.n_mpc_samples = n_mpc_samples
        self.n_final_samples = n_final_samples
        self.threshold = threshold
        self.max_per_segment = max_per_segment
        self.max_final_retries = max_final_retries
        self.save_buffer_details = save_buffer_details

        self.asker_model_name = asker_model_name
        self.split_mode = asker_model_name is not None

        self.lora_request = None
        if lora_path is not None:
            print(f"LoRA adapter will be loaded from: {lora_path}")
            self.lora_request = LoRARequest("askr_lora", 1, lora_path)

        print(f"Loading model with vLLM: {model_name}...")
        self.tokenizer = self._load_tokenizer(model_name, no_think, role="reasoner")
        self.asker_tokenizer = (
            self._load_tokenizer(asker_model_name, False, role="asker")
            if self.split_mode else self.tokenizer
        )

        if skip_vllm_init:
            print("Skipping vLLM initialization (will use external vLLM instance)")
            self.model = None
            self._asker_model = None
        else:
            if self.split_mode:
                # vLLM sizes each engine against total device memory, so the two utilizations must sum below 1.0.
                self._asker_model = self._build_llm(
                    asker_model_name,
                    tensor_parallel_size=asker_tensor_parallel_size,
                    gpu_memory_utilization=asker_gpu_memory_utilization,
                    max_model_len=asker_max_model_len or max_model_len,
                    lora_path=lora_path,
                    role="asker",
                )
                self.model = self._build_llm(
                    model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    lora_path=None,
                    role="reasoner",
                )
            else:
                self.model = self._build_llm(
                    model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    lora_path=lora_path,
                    role="model",
                )
                self._asker_model = None

    @property
    def asker_model(self):
        """The engine that generates sub-questions. Resolved at call time: the trainer assigns self.model after init."""
        return self._asker_model if self.split_mode else self.model

    def _load_tokenizer(self, model_name: str, no_think: bool, role: str = "model"):
        """Load a tokenizer; with no_think, force Qwen3's non-thinking chat template (enable_thinking=False)."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if no_think:
            _orig_template = tokenizer.apply_chat_template

            def _no_think_template(*a, **kw):
                kw.setdefault("enable_thinking", False)
                return _orig_template(*a, **kw)

            tokenizer.apply_chat_template = _no_think_template
            print(f"[askr] {role}: thinking mode DISABLED (enable_thinking=False on all chat templates)")

        return tokenizer

    def _build_llm(
        self,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        lora_path: Optional[str],
        role: str = "model",
    ) -> LLM:
        """Instantiate one vLLM engine."""
        vllm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": "bfloat16" if torch.cuda.is_available() else "float32",
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
        }

        if lora_path is not None:
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = 128

        llm = LLM(**vllm_kwargs)

        lora_str = f" with LoRA from {lora_path}" if lora_path else ""
        print(f"[askr] {role} loaded: {model_name}{lora_str} "
              f"(tensor_parallel_size={tensor_parallel_size}, "
              f"gpu_memory_utilization={gpu_memory_utilization}, max_model_len={max_model_len})")
        return llm

    def _question_types(self, batch_samples: Dict[str, List], dataset_type: Optional[str]) -> List[str]:
        """Per-question dataset type: batch_samples["types"] when given (mixed math/STEM data), else one type for all."""
        default_type = dataset_type or self.dataset_type
        types = batch_samples.get("types") or [None] * len(batch_samples["questions"])
        return [t or default_type for t in types]

    def _question_generation(
        self,
        batch_samples: Dict[str, List],
        n_samples_per_question: int = 1,
        dataset_type: Optional[str] = None,
        tmpc_mode: bool = False,
        tmpc_buffers: Optional[List[SegmentedBuffer]] = None,
        verbose: bool = True,
    ) -> List[str]:
        all_system_prompts = []
        all_user_prompts = []
        all_images_paths = []
        questions = batch_samples["questions"]
        question_types = self._question_types(batch_samples, dataset_type)
        options_list = batch_samples.get("options_list", [None] * len(questions))
        images_paths = batch_samples.get("image_paths", [None] * len(questions))
        num_questions = len(questions)
        total_samples = num_questions * n_samples_per_question

        for q_idx in range(num_questions):
            buffer_context = ""
            if tmpc_mode:
                buffer_context = tmpc_buffers[q_idx].get_prompt_context()
            for _ in range(n_samples_per_question):
                sys_p, usr_p = build_all_subq_generation_prompt(
                    questions[q_idx],
                    question_types[q_idx],
                    options=options_list[q_idx],
                    buffer_context=buffer_context,
                    max_subqs_len=self.max_subqs_len if self.max_subqs_len is not None else 10)
                all_system_prompts.append(sys_p)
                all_user_prompts.append(usr_p)
                all_images_paths.append(images_paths[q_idx] if images_paths else None)

        return generate_text_batch(
            all_system_prompts,
            all_user_prompts,
            total_samples,
            images_paths=all_images_paths if self.visual_mode else None,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tokenizer=self.asker_tokenizer,
            lora_request=self.lora_request,
            model=self.asker_model,
            enable_lora=True,
            verbose=verbose,
        )

    def solve_with_askr(
        self,
        batch_samples: Dict[str, List],
        start_sample_id: int = 1,
        askr_completions: Optional[List[str]] = None,
        verbose: bool = True,
        dataset_type: Optional[str] = None,
        tmpc_mode: bool = False,
        tmpc_buffers: Optional[List[SegmentedBuffer]] = None,
        n_samples_override: Optional[int] = None,
    ) -> List[List[Dict]]:
        """Generate sub-questions, answer them with the reasoner, extract and score the final answers,
        and (if enabled) self-verify every SubQA pair.

        Args:
            batch_samples: "questions", and optionally "ground_truths", "options_list", "image_paths",
                "types" (per-question dataset type)
            askr_completions: pre-generated sub-question chains, one per (question, sample), used in training
            n_samples_override: samples per question (defaults to self.n_samples)

        Returns:
            List of lists - outer list per question, inner list per sample
        """
        questions = batch_samples["questions"]
        question_types = self._question_types(batch_samples, dataset_type)
        ground_truths = batch_samples.get("ground_truths", None)
        options_list = batch_samples.get("options_list", None)
        images_paths = batch_samples.get("image_paths", None)
        n_samples_per_question = n_samples_override if n_samples_override is not None else self.n_samples

        if tmpc_mode:
            assert tmpc_buffers is not None and len(tmpc_buffers) == len(questions)

        num_questions = len(questions)
        total_samples = num_questions * n_samples_per_question

        if ground_truths is None:
            ground_truths = [None] * num_questions

        if options_list is None:
            options_list = [None] * num_questions

        if verbose:
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"Multi-Question AskR: {num_questions} questions × {n_samples_per_question} samples = {total_samples} total")
            tqdm.write(f"{'='*70}\n")

        start_time = time.time()

        sample_to_question = []
        for q_idx in range(num_questions):
            for s_idx in range(n_samples_per_question):
                sample_to_question.append((q_idx, s_idx))

        # Step 1: sub-questions
        if askr_completions is not None:
            assert len(askr_completions) == total_samples, (len(askr_completions), total_samples)
            all_subq_responses = askr_completions
        else:
            if verbose:
                tqdm.write(f"Step 1: Generating sub-questions for {total_samples} samples...")
            all_subq_responses = self._question_generation(
                batch_samples=batch_samples,
                n_samples_per_question=n_samples_per_question,
                dataset_type=dataset_type,
                tmpc_mode=tmpc_mode,
                tmpc_buffers=tmpc_buffers,
                verbose=verbose,
            )

        all_sub_questions = []
        for response in all_subq_responses:
            sub_questions = parse_sub_questions(response)
            if len(sub_questions) > self.max_subqs_len:
                sub_questions = sub_questions[:self.max_subqs_len]
            all_sub_questions.append(sub_questions)

        if verbose:
            for q_idx in range(num_questions):
                sample_indices = [i for i, (qidx, _) in enumerate(sample_to_question) if qidx == q_idx]
                avg_subqs = sum(len(all_sub_questions[i]) for i in sample_indices) / len(sample_indices) if sample_indices else 0
                tqdm.write(f"  Q{q_idx+1}: Avg {avg_subqs:.1f} sub-questions")

        # Step 2: answer all sub-questions and the original question
        all_subqa_pairs = [[] for _ in range(total_samples)]
        all_final_answer_text = [None] * total_samples
        all_contexts = [""] * total_samples

        if verbose:
            tqdm.write("\nStep 2: Answering all sub-questions")

        all_system_prompts = []
        all_user_prompts = []
        all_images_paths = []

        for sample_idx in range(total_samples):
            q_idx, _ = sample_to_question[sample_idx]
            sub_questions = all_sub_questions[sample_idx]
            image_p = images_paths[q_idx] if images_paths is not None else None

            if tmpc_mode:
                all_contexts[sample_idx] = tmpc_buffers[q_idx].get_prompt_context()

            if sub_questions:
                sys_p, usr_p = build_all_subqa_prompt(
                    question=questions[q_idx],
                    sub_questions=sub_questions,
                    buffer_context=all_contexts[sample_idx],
                    dataset_type=question_types[q_idx],
                    options=options_list[q_idx])
            else:
                sys_p, usr_p = "", ""

            all_system_prompts.append(sys_p)
            all_user_prompts.append(usr_p)
            all_images_paths.append(image_p)

        # The reasoner always answers without the asker's adapter.
        all_responses = generate_text_batch(
            all_system_prompts,
            all_user_prompts,
            total_samples,
            images_paths=all_images_paths if self.visual_mode else None,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tokenizer=self.tokenizer,
            lora_request=None if self.split_mode else self.lora_request,
            model=self.model,
            enable_lora=False,
            verbose=verbose,
        )

        for sample_idx, response in enumerate(all_responses):
            if not response:
                continue

            for sq, sa in parse_subqa_pairs(response):
                all_subqa_pairs[sample_idx].append({
                    "sub_question": sq,
                    "sub_answer": sa,
                    "verification_rate": 0 if self.do_verification else None,
                    "n_verify": self.n_verify if self.do_verification else 0,
                    "verification_details": []
                })
            all_final_answer_text[sample_idx] = response

        # Step 3: extract and score final answers
        if verbose:
            tqdm.write("\nStep 3: Extracting and verifying answers...")

        all_results = []
        verification_batch = []
        verification_indices = []

        for sample_idx in range(total_samples):
            q_idx, s_within_q = sample_to_question[sample_idx]
            question = questions[q_idx]
            ground_truth = ground_truths[q_idx]
            final_answer_text = all_final_answer_text[sample_idx]
            question_type = question_types[q_idx]

            extracted_answer = extract_boxed_answer(final_answer_text, question_type) if final_answer_text else None

            result_dict = {
                "question_idx": q_idx,
                "sample_id": start_sample_id + sample_idx,
                "sample_within_question": s_within_q + 1,
                "question": question,
                "ground_truth": ground_truth,
                "sub_questions": [pair["sub_question"] for pair in all_subqa_pairs[sample_idx]],
                "subqa_pairs": all_subqa_pairs[sample_idx],
                "final_answer_text": final_answer_text,
                "extracted_answer": extracted_answer,
                "is_correct": None,
                "verification_method": None,
                "time_elapsed": time.time() - start_time
            }

            if ground_truth and extracted_answer:
                if question_type == "stem":
                    verification_batch.append({
                        "prompt": question,
                        "llm_response": extracted_answer,
                        "gold_standard_solution": ground_truth,
                        "verification_info": {"judging_instructions": ""}
                    })
                    verification_indices.append(len(all_results))
                else:
                    is_correct, verification_method = verify_answer(extracted_answer, ground_truth, prompt=question, dataset_type=question_type)
                    result_dict["is_correct"] = is_correct
                    result_dict["verification_method"] = verification_method

            all_results.append(result_dict)

        if verification_batch:
            if verbose:
                tqdm.write(f"Verifying {len(verification_batch)} STEM answers concurrently...")
            verify_results = LlmJudgeVerifier(max_concurrent=32).verify_batch(verification_batch)
            for result_idx, verify_result in zip(verification_indices, verify_results):
                score = verify_result.get("score", 0)
                explanation = verify_result.get("verification_result_info", {}).get("judge_explanation", "No explanation")
                all_results[result_idx]["is_correct"] = score
                all_results[result_idx]["verification_method"] = f"STEM verification score: {score:.2f} | Explanation: {explanation[:100]}"

        # Step 4: the reasoner self-verifies every SubQA pair (all pairs of a trajectory in one call)
        if self.do_verification:
            if verbose:
                tqdm.write(f"\n{'='*70}")
                tqdm.write("All-at-Once Verification")
                tqdm.write(f"{'='*70}")
                tqdm.write(f"Samples: {total_samples} | Verification rounds: {self.n_verify}")

            verification_start = time.time()

            for i in range(self.n_verify):
                pass_start = time.time()
                system_prompts = []
                user_prompts = []
                all_images_paths = []

                for sample_idx in range(total_samples):
                    q_idx, _ = sample_to_question[sample_idx]
                    subqa_pairs = all_subqa_pairs[sample_idx]
                    image_p = images_paths[q_idx] if images_paths is not None else None

                    if not subqa_pairs:
                        system_prompts.append("")
                        user_prompts.append("")
                        all_images_paths.append(image_p)
                        continue

                    subqa_list = [
                        {"sub_question": pair["sub_question"], "sub_answer": pair["sub_answer"]}
                        for pair in subqa_pairs
                    ]
                    sys_p, usr_p = build_batch_verification_prompt(
                        questions[q_idx],
                        subqa_list,
                        dataset_type=question_types[q_idx],
                        options=options_list[q_idx]
                    )
                    system_prompts.append(sys_p)
                    user_prompts.append(usr_p)
                    all_images_paths.append(image_p)

                # Self-verification runs on the frozen reasoner, never the adapter.
                verify_responses = generate_text_batch(
                    system_prompts,
                    user_prompts,
                    total_samples,
                    images_paths=all_images_paths if self.visual_mode else None,
                    max_tokens=self.max_tokens,
                    tokenizer=self.tokenizer,
                    temperature=self.temperature,
                    model=self.model,
                    verbose=verbose,
                )

                for sample_idx, response in enumerate(verify_responses):
                    num_subqas = len(all_subqa_pairs[sample_idx])
                    if num_subqas == 0:
                        continue

                    verdicts, reasoning = parse_batch_verification_response(response, num_subqas)

                    for j in range(num_subqas):
                        all_subqa_pairs[sample_idx][j]["verification_rate"] += 1 if verdicts[j] else 0
                        all_subqa_pairs[sample_idx][j]["verification_details"].append({
                            "is_verified": verdicts[j],
                            "reasoning": reasoning
                        })

                if verbose:
                    tqdm.write(f"  Pass [{i + 1}/{self.n_verify}] completed in {time.time() - pass_start:.2f}s")

            for sample_idx in range(total_samples):
                for pair in all_subqa_pairs[sample_idx]:
                    pair["verification_rate"] = pair["verification_rate"] / self.n_verify
                all_results[sample_idx]["subqa_pairs"] = all_subqa_pairs[sample_idx]

            if verbose:
                tqdm.write(f"\n{'='*70}")
                tqdm.write(f"Verification completed in {time.time() - verification_start:.2f}s")
                tqdm.write(f"{'='*70}")

        results_by_question = [[] for _ in range(num_questions)]
        for result in all_results:
            results_by_question[result["question_idx"]].append(result)

        if verbose:
            tqdm.write(f"\n{'='*70}")
            tqdm.write("Results Summary:")
            for q_idx in range(num_questions):
                correct_count = sum(r["is_correct"] for r in results_by_question[q_idx]
                                if r["is_correct"] is not None)
                tqdm.write(f"  Q{q_idx+1}: {correct_count}/{n_samples_per_question} samples correct")
            tqdm.write(f"\nMulti-question batch completed in {time.time() - start_time:.2f}s")
            tqdm.write(f"{'='*70}\n")

        return results_by_question

    def solve_with_tmpc(
        self,
        batch_samples: Dict[str, List],
    ) -> Tuple[List[List[Dict]], List[SegmentedBuffer]]:
        """AskR with TMPC. Selection uses only the intrinsic reward; ground truth is never read here.

        Returns:
            (results, buffers): one selected sample per question, and each question's SegmentedBuffer
        """
        questions = batch_samples["questions"]
        buffers = [SegmentedBuffer(threshold=self.threshold, max_per_segment=self.max_per_segment) for _ in questions]

        for iteration_id in range(1, self.n_iterations + 1):
            result_by_question = self.solve_with_askr(
                batch_samples,
                tmpc_mode=True,
                tmpc_buffers=buffers,
                verbose=True,
                n_samples_override=self.n_mpc_samples,
            )

            for q_idx, result in enumerate(result_by_question):
                for s_idx, sample in enumerate(result):
                    sample_id = f"iter{iteration_id}_q{q_idx+1}_s{s_idx+1}"
                    subqa_pairs = sample.get("subqa_pairs", [])
                    trajectory = TrajectoryEntry(
                        trajectory_id=sample_id,
                        subqa_entries=[
                            SubQAEntry(
                                sub_question=pair["sub_question"],
                                sub_answer=pair["sub_answer"],
                                reward=pair.get("verification_rate", 0),
                                position=j,
                                trajectory_id=sample_id
                            ) for j, pair in enumerate(subqa_pairs)
                        ],
                        avg_reward=intrinsic_reward(sample),
                    )
                    buffers[q_idx].add_trajectory(trajectory)

            for buffer in buffers:
                buffer.save_history_snapshot(iteration_id)

        # Final rollout: per question, take the first candidate whose reward reaches the best reward in its buffer.
        num_questions = len(questions)
        target_rewards = [b.best_reward for b in buffers]
        max_retries = self.max_final_retries
        selected_per_question: List[Optional[List[Dict]]] = [None] * num_questions
        final_retry_count = 0
        final_candidates = []

        tqdm.write(f"\n--- Final Rollout (n={self.n_final_samples}, per-question target reward, "
                   f"max retries: {max_retries}) ---")

        while final_retry_count < max_retries and any(s is None for s in selected_per_question):
            final_retry_count += 1

            final_results = self.solve_with_askr(
                batch_samples,
                tmpc_mode=True,
                tmpc_buffers=buffers,
                verbose=True,
                n_samples_override=self.n_final_samples,
            )
            final_candidates.append(final_results)

            for q_idx, question_results in enumerate(final_results):
                if selected_per_question[q_idx] is not None:
                    continue
                for sample in question_results:
                    if intrinsic_reward(sample) >= target_rewards[q_idx]:
                        selected_per_question[q_idx] = [sample]
                        break

            n_met = sum(s is not None for s in selected_per_question)
            tqdm.write(f"  Final rollout {final_retry_count}/{max_retries}: "
                       f"{n_met}/{num_questions} questions met their target reward")

        def _best_candidate(q_idx: int) -> List[Dict]:
            """Highest-reward candidate for this question, as a 1-list."""
            candidates = [s for retry_results in final_candidates for s in retry_results[q_idx]]
            return [max(candidates, key=intrinsic_reward)] if candidates else []

        n_fallback = sum(1 for s in selected_per_question if s is None)
        if n_fallback:
            tqdm.write(f"--- Fallback: best-reward candidate for {n_fallback}/{num_questions} "
                       f"questions that never met target ---")
        final_results = [
            selected_per_question[q_idx] if selected_per_question[q_idx] is not None
            else _best_candidate(q_idx)
            for q_idx in range(num_questions)
        ]

        return final_results, buffers

    def batch_evaluate(
        self,
        dataset_samples: List[Dict],
        output_file: Optional[str] = None,
        batch_size: int = 1,
        save_details: bool = True,
        output_format: str = "json",
    ) -> Dict:
        """Evaluate on dataset samples and compute accuracy.

        Accuracy counts every sample, so an answer that cannot be parsed counts as wrong.

        Args:
            dataset_samples: samples with 'prompt' and 'answer' keys
            output_file: path to save results
            batch_size: number of QUESTIONS processed in parallel
            save_details: also save SubQA trajectories and response text
            output_format: 'json', 'json.gz', 'jsonl' or 'jsonl.gz'
        """
        results = []
        correct_count = 0
        scorable_count = 0
        attempted_count = 0
        samples_per_question = self.n_samples if self.mode == "askr" else 1

        tqdm.write(f"\n{'#'*70}")
        tqdm.write(f"Batch Evaluation: {len(dataset_samples)} questions | mode: {self.mode}")
        if self.mode == "tmpc":
            tqdm.write(f"TMPC: {self.n_iterations} iterations × {self.n_mpc_samples} samples, "
                       f"final rollout {self.n_final_samples} samples")
        else:
            tqdm.write(f"N samples per question: {self.n_samples}")
        tqdm.write(f"Question batch size: {batch_size}")
        tqdm.write(f"Max SubQ steps: {self.max_subqs_len}")
        tqdm.write(f"{'#'*70}\n")

        num_batches = (len(dataset_samples) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            batch_samples = dataset_samples[start_idx:start_idx + batch_size]

            batch = {
                "questions": [s.get('prompt', s.get('question', '')) for s in batch_samples],
                "ground_truths": [s.get('answer', s.get('gold_standard_solution', None)) for s in batch_samples],
                "options_list": [s.get('options', None) for s in batch_samples],
                "image_paths": [s.get('image_paths', []) for s in batch_samples],
                "metadata": [s.get('metadata', {}) for s in batch_samples],
            }

            tqdm.write(f"\n{'#'*70}")
            tqdm.write(f"Processing Question Batch {batch_idx + 1}/{num_batches} ({len(batch['questions'])} questions)")
            tqdm.write(f"{'#'*70}\n")

            batch_buffers = None
            if self.mode == "tmpc":
                batch_results_by_question, batch_buffers = self.solve_with_tmpc(batch)
            else:
                batch_results_by_question = self.solve_with_askr(batch)

            for q_idx, samples in enumerate(batch_results_by_question):
                question_correct = 0
                attempted_count += len(samples)
                for s in samples:
                    if s['is_correct'] is not None:
                        scorable_count += 1
                        if s['is_correct'] > 0.5:
                            question_correct += 1
                            correct_count += 1

                question_accuracy = question_correct / len(samples) if samples else 0.0

                sample_entries = []
                for s in samples:
                    entry = {
                        "sample_id": s["sample_id"],
                        "extracted_answer": s.get("extracted_answer", ""),
                        "is_correct": s.get("is_correct", 0.0),
                        "time_elapsed": s.get("time_elapsed", 0.0),
                    }
                    if save_details:
                        entry.update({
                            "sub_questions": s.get("sub_questions", []),
                            "subqa_pairs": s.get("subqa_pairs", []),
                            "final_answer_text": s.get("final_answer_text", ""),
                            "verification_method": s.get("verification_method", ""),
                        })
                    sample_entries.append(entry)

                result_entry = {
                    "question": batch["questions"][q_idx],
                    "ground_truth": batch["ground_truths"][q_idx],
                    "n_samples": len(samples),
                    "correct_samples": question_correct,
                    "question_accuracy": question_accuracy,
                    "question_metadata": batch["metadata"][q_idx],
                    "samples": sample_entries,
                }
                if batch_buffers is not None and self.save_buffer_details:
                    result_entry["buffer_history"] = batch_buffers[q_idx].history

                results.append(result_entry)

                tqdm.write(f"\nQuestion {start_idx + q_idx + 1} Accuracy: {question_correct}/{len(samples)} = {question_accuracy:.1%}")

        overall_accuracy = correct_count / attempted_count if attempted_count > 0 else 0.0

        summary = {
            "total_questions": len(dataset_samples),
            "n_samples_per_question": samples_per_question,
            "total_samples": attempted_count,
            "unparseable_samples": attempted_count - scorable_count,
            "correct_samples": correct_count,
            "overall_accuracy": overall_accuracy,
            "model": self.model_name,
            "asker_model": self.asker_model_name,
            "configuration": {
                "batch_size": batch_size,
                "mode": self.mode,
                "split_mode": self.split_mode,
                "n_verify": self.n_verify,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "save_details": save_details,
                "dataset_type": self.dataset_type,
                "max_model_len": self.max_model_len,
                "max_subqs_len": self.max_subqs_len,
                "lora_path": self.lora_path,
                "n_iterations": self.n_iterations,
                "n_mpc_samples": self.n_mpc_samples,
                "n_final_samples": self.n_final_samples,
                "threshold": self.threshold,
                "max_per_segment": self.max_per_segment,
                "max_final_retries": self.max_final_retries,
                "save_buffer_details": self.save_buffer_details,
            },
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if output_file:
            saved_path = save_results(summary, output_file, output_format)
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"Results saved to: {saved_path}")
            tqdm.write(f"{'='*70}\n")

        tqdm.write(f"\n{'='*70}")
        tqdm.write("EVALUATION SUMMARY")
        tqdm.write(f"{'='*70}")
        tqdm.write(f"Total Questions: {len(dataset_samples)}")
        tqdm.write(f"Samples per question: {samples_per_question}")
        tqdm.write(f"Total Samples: {attempted_count} ({attempted_count - scorable_count} unparseable, counted as wrong)")
        tqdm.write(f"Correct Samples: {correct_count}")
        tqdm.write(f"Overall Accuracy: {overall_accuracy:.2%}")
        tqdm.write(f"{'='*70}\n")

        return summary


def main():
    parser = argparse.ArgumentParser(description="AskR inference: AskR and AskR + TMPC")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Reasoner (and, without --asker-model, the asker)")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=EVAL_DATASETS)
    parser.add_argument("--dataset-samples", type=int, default=-1, help="Number of questions to evaluate (-1 = all)")
    parser.add_argument("--mode", type=str, default="tmpc", choices=["askr", "tmpc"])
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--n-samples", type=int, default=8, help="Samples per question in --mode askr")
    parser.add_argument("--n-verify", type=int, default=1, help="Self-verification rounds per SubQ-SubA pair")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of questions processed in parallel")
    parser.add_argument("--max-subq-steps", type=int, default=10, help="Maximum number of sub-questions per trajectory")
    parser.add_argument("--do-verify", action="store_true",
                        help="Self-verify SubQ-SubA pairs in --mode askr (always on in tmpc)")
    parser.add_argument("--no-think", action="store_true",
                        help="Force enable_thinking=False (Qwen3 hybrid models)")
    parser.add_argument("--visual-mode", action="store_true", help="Vision-language benchmarks (Qwen-VL models)")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: outputs/eval/<auto-generated name>)")
    parser.add_argument("--output-format", type=str, default="json", choices=["json", "json.gz", "jsonl", "jsonl.gz"])
    parser.add_argument("--no-save-details", action="store_true", help="Only save answers and accuracy, not SubQA trajectories")
    parser.add_argument("--no-save-buffer-details", action="store_true", help="Don't save TMPC buffer history")

    # TMPC
    parser.add_argument("--n-iterations", type=int, default=3)
    parser.add_argument("--n-mpc-samples", type=int, default=8, help="Candidates per TMPC iteration")
    parser.add_argument("--n-final-samples", type=int, default=8, help="Candidates in the final rollout")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum intrinsic reward to enter the buffer")
    parser.add_argument("--max-per-segment", type=int, default=3, help="SubQA pairs kept per buffer segment")
    parser.add_argument("--max-final-retries", type=int, default=1, help="Final rollouts to try before falling back to the best-reward candidate")

    # vLLM
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=16384)

    parser.add_argument("--lora-path", type=str, default=None, help="Trained AskR adapter (omit for the untrained asker)")

    # Separate asker / reasoner (reasoner-scale experiments)
    parser.add_argument("--asker-model", type=str, default=None,
                        help="Separate asker model (e.g. Qwen/Qwen3-4B). --lora-path is applied to it and --model "
                             "only answers and verifies. --asker-gpu-memory-utilization + "
                             "--gpu-memory-utilization must stay below ~0.9.")
    parser.add_argument("--asker-gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--asker-max-model-len", type=int, default=None, help="Defaults to --max-model-len")
    parser.add_argument("--asker-tensor-parallel-size", type=int, default=1)

    args = parser.parse_args()

    if args.output is None:
        lora_suffix = "_lora" if args.lora_path else ""
        filename = f"askr_{args.mode}{lora_suffix}_{args.dataset}_q{args.dataset_samples}_steps{args.max_subq_steps}.json"
        args.output = os.path.join("outputs", "eval", filename)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    dataset_samples = load_dataset_by_name(args.dataset, args.dataset_samples)

    solver = AskRSolver(
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n_samples=args.n_samples,
        n_verify=args.n_verify,
        do_verification=args.do_verify,
        dataset_type=args.dataset,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_subqs_len=args.max_subq_steps,
        max_model_len=args.max_model_len,
        lora_path=args.lora_path,
        visual_mode=args.visual_mode,
        no_think=args.no_think,
        mode=args.mode,
        asker_model_name=args.asker_model,
        asker_gpu_memory_utilization=args.asker_gpu_memory_utilization,
        asker_max_model_len=args.asker_max_model_len,
        asker_tensor_parallel_size=args.asker_tensor_parallel_size,
        n_iterations=args.n_iterations,
        n_mpc_samples=args.n_mpc_samples,
        n_final_samples=args.n_final_samples,
        threshold=args.threshold,
        max_per_segment=args.max_per_segment,
        max_final_retries=args.max_final_retries,
        save_buffer_details=not args.no_save_buffer_details,
    )

    solver.batch_evaluate(
        dataset_samples,
        args.output,
        batch_size=args.batch_size,
        save_details=not args.no_save_details,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
