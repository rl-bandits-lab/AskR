# This code base is based on Genesys (https://github.com/PrimeIntellect-ai/genesys)
import os
import asyncio
from typing import Dict, List
from openai import AsyncOpenAI
from .utils import extract_json
from tqdm.asyncio import tqdm as async_tqdm

JUDGE_PROMPT = """
Your job is to judge the output of a large language model trained to reason about hard problems such as mathematics, science, and coding. Below you will see a problem given to the model, a gold standard response provided by a human, and the model's provided response.
Your job is to grade the model response's correctness by how close it is to the content of the gold standard response on a scale of 1-100. The rating should NOT be about the similarity of the style, but about whether the llm response is given the same factual statements as the gold standard response.
The grading should be quite strict, you shouldn't just give every response a score of 70 or better. Only give high scores if the model actually matches the gold standard response.
{additional_judging_instructions}
Here is the problem:
{problem}

---

Here is the gold standard response:
{gold_standard_response}

---

Here is the response the model provided:
{llm_response}

---

Think about your score and then return a json with a first field 'explanation' explaining your score and a second field 'score' indicating how close the content of the LLM response is to the human response on a scale of 1-100.
"""


class LlmJudgeVerifier:
    def __init__(self, max_concurrent: int = 32):
        self.max_concurrent = max_concurrent
        self.semaphore = None

    async def verify_async(self, result: Dict, provider: str, model_name: str) -> Dict:
        """Score one response against its gold standard. Returns {"score": 0-1, "verification_result_info": {...}}."""
        async with self.semaphore:
            client = self._init_async_client(provider)
            prompt = JUDGE_PROMPT.format(
                problem=result["prompt"],
                gold_standard_response=result["gold_standard_solution"],
                llm_response=result["llm_response"],
                additional_judging_instructions=self._build_additional_judging_instructions(
                    result["verification_info"]["judging_instructions"]
                ),
            )

            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                gen_json = extract_json(response.choices[0].message.content)
                return dict(
                    score=gen_json["score"] / 100,
                    verification_result_info=dict(judge_explanation=gen_json["explanation"])
                )
            except Exception as e:
                return dict(
                    score=0.0,
                    verification_result_info=dict(judge_explanation=f"Error: {str(e)}")
                )

    def verify_batch(self, results: List[Dict]) -> List[Dict]:
        """Score many responses concurrently. Output order matches `results`."""
        provider = os.environ.get("LLM_JUDGE_PROVIDER", "deepinfra")
        if provider == "openai":
            model_name = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o")
        else:
            model_name = os.environ.get("LLM_JUDGE_MODEL", "deepseek-ai/DeepSeek-V3.2")
        return asyncio.run(self._verify_batch_async(results, provider, model_name))

    async def _verify_batch_async(self, results: List[Dict], provider: str, model_name: str) -> List[Dict]:
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self.verify_async(result, provider, model_name) for result in results]
        return await async_tqdm.gather(*tasks, desc="Verifying")

    def _init_async_client(self, provider: str) -> AsyncOpenAI:
        if provider == "deepinfra":
            api_key = os.environ.get("DEEPINFRA_API_KEY")
            if not api_key:
                raise ValueError("DEEPINFRA_API_KEY environment variable not set")
            return AsyncOpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return AsyncOpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: deepinfra, openai")

    def _build_additional_judging_instructions(self, judging_instructions: str) -> str:
        if judging_instructions == "":
            return ""
        return f"\nHere are additional judging instructions:\n{judging_instructions}\n\n"
