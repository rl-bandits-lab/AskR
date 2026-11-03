from typing import List, Tuple, Dict


def build_all_subq_generation_prompt(
          question: str,
          dataset_type: str,
          options: List[str] = None,
          max_subqs_len: int = 10,
          buffer_context: str = "",
          ) -> Tuple[str, str]:
        """Build prompt for generating chain of sub-questions."""
        buffer_msg = f"Use the following context to help generate sub-questions:\n{buffer_context}" if buffer_context else ""
        if dataset_type in ["vstar", "cvbench", "mmstar", "lpt30k"]:
            options_text = "Options:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) if options else ""

            system_prompt = """You are a helpful assistant that breaks down complex visual reasoning multiple-choice questions into simpler sub-questions.
When given a question with an image and options, decompose it into a chain of sub-questions that, when answered sequentially, will help analyze the visual information and determine the correct answer."""

            user_prompt = f"""Break down the following visual reasoning multiple-choice question into a chain of sub-questions. An image is provided that contains essential information. Each sub-question should:
1. Be simpler and more focused than the original question
2. Include sub-questions about analyzing the visual content of the image (e.g., "What objects/features are visible in the image?")
3. Build upon previous sub-questions
4. Lead step-by-step toward determining the correct answer choice
5. Help analyze and evaluate the given options based on visual evidence

{buffer_msg}

Question: {question}

{options_text}

Please generate N sub-questions in the following format, you should decide how many N by yourself (But N <= {max_subqs_len}):
SubQ1: [First sub-question - often about visual observation]
SubQ2: [Second sub-question]
SubQ3: [Third sub-question]
...

Only output the sub-questions, one per line, in the format "SubQN: [question]"."""
        elif dataset_type in ["mmlu", "lab-bench"]:
            options_text = "Options:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) if options else ""

            system_prompt = """You are a helpful assistant that breaks down complex multiple-choice questions into simpler sub-questions.
When given a question with options, decompose it into a chain of sub-questions that, when answered sequentially, will help determine the correct answer."""

            user_prompt = f"""Break down the following multiple-choice question into a chain of sub-questions. Each sub-question should:
1. Be simpler and more focused than the original question
2. Build upon previous sub-questions
3. Lead step-by-step toward determining the correct answer choice
4. Help analyze and evaluate the given options

{buffer_msg}

Question: {question}

{options_text}

Please generate N sub-questions in the following format, you should decide how many N by yourself (But N <= {max_subqs_len}):
SubQ1: [First sub-question]
SubQ2: [Second sub-question]
SubQ3: [Third sub-question]
...

Only output the sub-questions, one per line, in the format "SubQN: [question]"."""
        elif dataset_type == "scieval":
            system_prompt = """You are a helpful assistant that breaks down complex multiple-choice questions into simpler sub-questions.
When given a question with options, decompose it into a chain of sub-questions that, when answered sequentially, will help determine the correct answer."""

            user_prompt = f"""Break down the following multiple-choice question into a chain of sub-questions. Each sub-question should:
1. Be simpler and more focused than the original question
2. Build upon previous sub-questions
3. Lead step-by-step toward determining the correct answer choice
4. Help analyze and evaluate the given options

{buffer_msg}

Question: {question}

Please generate N sub-questions in the following format, you should decide how many N by yourself (But N <= {max_subqs_len}):
SubQ1: [First sub-question]
SubQ2: [Second sub-question]
SubQ3: [Third sub-question]
...

Only output the sub-questions, one per line, in the format "SubQN: [question]"."""
        elif dataset_type == "stem":
            system_prompt = """You are a helpful assistant that breaks down complex technical questions into simpler sub-questions.
When given a question from a technical forum, decompose it into a chain of sub-questions that, when answered sequentially, will help provide a comprehensive and informative response."""

            user_prompt = f"""Break down the following technical question into a chain of sub-questions. Each sub-question should:
1. Be simpler and more focused than the original question
2. Build upon previous sub-questions
3. Lead step-by-step toward a complete, helpful answer
4. Be answerable independently

{buffer_msg}

Question: {question}

Please generate N sub-questions in the following format, you should decide how many N by yourself (But N <= {max_subqs_len}):
SubQ1: [First sub-question]
SubQ2: [Second sub-question]
SubQ3: [Third sub-question]
SubQ4: [Fourth sub-question]
SubQ5: [Fifth sub-question]
...

Only output the sub-questions, one per line, in the format "SubQN: [question]"."""
        else:
            system_prompt = """You are a helpful assistant that breaks down complex problems into simpler sub-questions.
When given a problem, decompose it into a chain of sub-questions that, when answered sequentially, will lead to the final answer."""

            user_prompt = f"""Break down the following problem into a chain of sub-questions. Each sub-question should:
1. Be simpler than the original problem
2. Build upon previous sub-questions
3. Lead step-by-step toward the final answer
4. Be answerable independently

{buffer_msg}

Problem: {question}

Please generate N sub-questions in the following format, you should decide how many N by yourself (But N <= {max_subqs_len}):
SubQ1: [First sub-question]
SubQ2: [Second sub-question]
SubQ3: [Third sub-question]
SubQ4: [Fourth sub-question]
SubQ5: [Fifth sub-question]
...

Only output the sub-questions, one per line, in the format "SubQN: [question]"."""

        return system_prompt, user_prompt
    
def build_all_subqa_prompt(
          question: str,
          sub_questions: List[str],
          context: str = "",
          buffer_context: str = "",
          dataset_type: str = "math",
          options: List[str] = None) -> Tuple[str, str]:
        """Build prompt for answering ALL sub-questions one by one, then providing the final answer.

        This combines the approach of build_subqa_prompt and build_final_answer_prompt:
        - Answer each sub-question sequentially
        - Use previous answers as context for subsequent sub-questions
        - Finally synthesize all sub-answers to provide the final answer

        Args:
            question: The original root question to solve
            sub_questions: List of sub-questions to answer sequentially
            context: Optional initial context
            options: For MMLU, the list of answer choices

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        context = f"{buffer_context}\n{context}".strip() if buffer_context else context.strip()
        if dataset_type in ["vstar", "cvbench", "mmstar", "lpt30k"]:
            options_text = "Options:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) if options else ""

            system_prompt = """You are a helpful assistant that solves visual reasoning multiple-choice questions by systematically analyzing sub-questions.
Carefully examine the provided image, answer each sub-question clearly using visual evidence, then use all the sub-answers to determine which option is correct."""

            context_text = f"\n\nInitial Context:\n{context}" if context else ""

            subq_list = "\n".join([f"{i+1}. {sq}" for i, sq in enumerate(sub_questions)])

            user_prompt = f"""Original Question: {question}

{options_text}
{context_text}

An image is provided that contains essential visual information for answering this question.

To determine the correct answer, answer the following sub-questions one by one. Each answer should be clear and analytical, referencing visual observations from the image when relevant.

Sub-Questions:
{subq_list}

Please provide your response in the following format:

SubQ1: [First sub-question]
SubA1: [Your analysis for the first sub-question, including visual observations if applicable]

SubQ2: [Second sub-question]
SubA2: [Your analysis for the second sub-question, using information from SubA1 and the image if needed]

SubQ3: [Third sub-question]
SubA3: [Your analysis for the third sub-question, using information from previous answers and visual evidence if needed]

... (continue for all sub-questions)

After answering all sub-questions, determine the correct answer choice based on the visual evidence and your analysis:

Final Answer: [Single letter A-J indicating your chosen answer]"""
        elif dataset_type in ["mmlu", "lab-bench", "scieval"]:
            options_text = "Options:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) if options else ""

            system_prompt = """You are a helpful assistant that solves multiple-choice questions by systematically analyzing sub-questions.
Answer each sub-question clearly, then use all the sub-answers to determine which option is correct."""

            context_text = f"\n\nInitial Context:\n{context}" if context else ""

            subq_list = "\n".join([f"{i+1}. {sq}" for i, sq in enumerate(sub_questions)])

            user_prompt = f"""Original Question: {question}

{options_text}
{context_text}

To determine the correct answer, answer the following sub-questions one by one. Each answer should be clear and analytical.

Sub-Questions:
{subq_list}

Please provide your response in the following format:

SubQ1: [First sub-question]
SubA1: [Your analysis for the first sub-question]

SubQ2: [Second sub-question]
SubA2: [Your analysis for the second sub-question, using information from SubA1 if needed]

SubQ3: [Third sub-question]
SubA3: [Your analysis for the third sub-question, using information from previous answers if needed]

... (continue for all sub-questions)

After answering all sub-questions, determine the correct answer choice:

Final Answer: [Single letter A-J indicating your chosen answer]"""
        elif dataset_type == "stem":
            system_prompt = """You are a helpful technical expert that answers complex questions by breaking them down into sub-questions.
Answer each sub-question clearly and informatively, then use all the sub-answers to provide a comprehensive final response."""

            context_text = f"\n\nInitial Context:\n{context}" if context else ""

            subq_list = "\n".join([f"{i+1}. {sq}" for i, sq in enumerate(sub_questions)])

            user_prompt = f"""Original Question: {question}
{context_text}

To provide a complete answer to this question, answer the following sub-questions one by one. Each answer should be clear and informative.

Sub-Questions:
{subq_list}

Please provide your response in the following format:

SubQ1: [First sub-question]
SubA1: [Your answer to the first sub-question]

SubQ2: [Second sub-question]
SubA2: [Your answer to the second sub-question, using information from SubA1 if needed]

SubQ3: [Third sub-question]
SubA3: [Your answer to the third sub-question, using information from previous answers if needed]

... (continue for all sub-questions)

After answering all sub-questions, synthesize the information to provide the final comprehensive response:

Final Answer: [Your complete, helpful response to the original question]"""
        elif dataset_type == "math":
            system_prompt = """You are a helpful assistant that solves complex problems by breaking them down into sub-questions.
Answer each sub-question clearly and concisely, then use all the sub-answers to provide the final answer."""

            context_text = f"\n\nInitial Context:\n{context}" if context else ""

            subq_list = "\n".join([f"{i+1}. {sq}" for i, sq in enumerate(sub_questions)])

            user_prompt = f"""Original Problem: {question}
{context_text}

To solve this problem, answer the following sub-questions one by one. Each answer should be clear and concise.

Sub-Questions:
{subq_list}

Please provide your response in the following format:

SubQ1: [First sub-question]
SubA1: [Your answer to the first sub-question]

SubQ2: [Second sub-question]
SubA2: [Your answer to the second sub-question, using information from SubA1 if needed]

SubQ3: [Third sub-question]
SubA3: [Your answer to the third sub-question, using information from previous answers if needed]

... (continue for all sub-questions)

After answering all sub-questions, synthesize the information to provide the final answer:

Final Answer: \\boxed{{<answer>}}"""
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        return system_prompt, user_prompt

def build_batch_verification_prompt(original_question: str, subqa_pairs: List[Dict], dataset_type: str = "math", options: List[str] = None) -> Tuple[str, str]:
        """Build prompt for verifying ALL SubQ-SubA pairs at once (non-Markovian).

        This verification evaluates all sub-questions and sub-answers simultaneously,
        allowing the model to assess their collective correctness and consistency.

        Args:
            original_question: The original question
            subqa_pairs: List of dicts with 'sub_question' and 'sub_answer' keys

        Returns:
            (system_prompt, user_prompt)
        """
        if dataset_type in ["vstar", "cvbench", "mmstar", "lpt30k"]:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) if options else ""

            system_prompt = """You are an expert at verifying the correctness of reasoning chains for visual reasoning multiple-choice questions.
Your task is to evaluate whether each sub-question and sub-answer pair contributes meaningfully to determining the correct answer choice, considering both visual observations and logical reasoning."""

            subqa_text = ""
            for i, pair in enumerate(subqa_pairs, 1):
                subqa_text += f"\n{i}. Sub-Question: {pair['sub_question']}\n   Sub-Answer: {pair['sub_answer']}\n"

            user_prompt = f"""Original Question: {original_question}

Options:
{options_text}

Note: An image was provided for this question that contains essential visual information.

Reasoning Chain (All SubQ-SubA pairs):
{subqa_text}

Task: Evaluate the correctness of EACH sub-answer in the context of the entire reasoning chain and the visual information from the image.

For each SubQ-SubA pair, assess:
1. **Relevance**: Does the sub-answer directly address its sub-question?
2. **Visual Accuracy**: If the sub-answer involves visual observations, are they accurate based on the image?
3. **Correctness**: Is the sub-answer factually accurate in analyzing the options?
4. **Completeness**: Does the sub-answer provide sufficient information?
5. **Consistency**: Is the sub-answer consistent with other sub-answers in the chain?

Provide your analysis in the following format:

REASONING:
For each SubQ-SubA pair (1, 2, 3, ...), analyze its correctness considering the entire chain, the visual evidence, and the answer choices.

VERDICTS:
Provide one verdict per line in the exact format:
SubQ 1: [true/false]
SubQ 2: [true/false]
SubQ 3: [true/false]
...

Where "true" means the sub-answer is correct and "false" means it is incorrect or problematic."""
        elif dataset_type in ["mmlu", "lab-bench"]:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) if options else ""

            system_prompt = """You are an expert at verifying the correctness of reasoning chains for multiple-choice questions.
Your task is to evaluate whether each sub-question and sub-answer pair contributes meaningfully to determining the correct answer choice."""

            subqa_text = ""
            for i, pair in enumerate(subqa_pairs, 1):
                subqa_text += f"\n{i}. Sub-Question: {pair['sub_question']}\n   Sub-Answer: {pair['sub_answer']}\n"

            user_prompt = f"""Original Question: {original_question}

Options:
{options_text}

Reasoning Chain (All SubQ-SubA pairs):
{subqa_text}

Task: Evaluate the correctness of EACH sub-answer in the context of the entire reasoning chain.

For each SubQ-SubA pair, assess:
1. **Relevance**: Does the sub-answer directly address its sub-question?
2. **Correctness**: Is the sub-answer factually accurate in analyzing the options?
3. **Completeness**: Does the sub-answer provide sufficient information?
4. **Consistency**: Is the sub-answer consistent with other sub-answers in the chain?

Provide your analysis in the following format:

REASONING:
For each SubQ-SubA pair (1, 2, 3, ...), analyze its correctness considering the entire chain and the answer choices.

VERDICTS:
Provide one verdict per line in the exact format:
SubQ 1: [true/false]
SubQ 2: [true/false]
SubQ 3: [true/false]
...

Where "true" means the sub-answer is correct and "false" means it is incorrect or problematic."""
        elif dataset_type == "stem":
            system_prompt = """You are an expert at verifying the correctness and quality of reasoning chains for technical questions.
Your task is to evaluate whether each sub-question and sub-answer pair contributes meaningfully to creating a high-quality, comprehensive final answer."""

            subqa_text = ""
            for i, pair in enumerate(subqa_pairs, 1):
                subqa_text += f"\n{i}. Sub-Question: {pair['sub_question']}\n   Sub-Answer: {pair['sub_answer']}\n"

            user_prompt = f"""Original Question: {original_question}

Reasoning Chain (All SubQ-SubA pairs):
{subqa_text}

Task: For EACH SubQ-SubA pair, evaluate TWO critical aspects:

A) **Individual Correctness**: Is the sub-answer accurate, complete, and technically sound?
B) **Contribution to Final Answer Quality**: Did this SubQ-SubA pair contribute to achieving a high-quality final answer?

For each SubQ-SubA pair, analyze:

**Quality Impact**: Given the final answer quality was UNKNOWN", did this SubQ-SubA pair:
   - Provide essential information that improved the final answer?
   - Add valuable context or explanation?
   - Help structure the reasoning in a clear, logical way?
   - OR did it introduce errors, irrelevant information, or confusion that degraded quality?

Provide your analysis in the following format:

REASONING:
For each SubQ-SubA pair (1, 2, 3, ...), explain:
- Whether the sub-answer is correct
- Whether it contributed positively to the final answer quality
- How it impacted the overall response (positive, neutral, or negative contribution)

VERDICTS:
Provide one verdict per line in the exact format:
SubQ 1: [true/false]
SubQ 2: [true/false]
SubQ 3: [true/false]
...

Where:
- "true" means the sub-answer is BOTH correct AND contributed positively to achieving a high-quality final answer
- "false" means the sub-answer is either incorrect OR did not contribute meaningfully to the final answer quality (even if technically correct, it may be irrelevant or unhelpful)"""
        else:

            system_prompt = """You are an expert at verifying the correctness of reasoning chains.
You will evaluate all sub-questions and sub-answers simultaneously to assess their collective quality."""

            subqa_text = ""
            for i, pair in enumerate(subqa_pairs, 1):
                subqa_text += f"\n{i}. Sub-Question: {pair['sub_question']}\n   Sub-Answer: {pair['sub_answer']}\n"

            user_prompt = f"""Original Problem: {original_question}

Reasoning Chain (All SubQ-SubA pairs):
{subqa_text}

Task: Evaluate the correctness of EACH sub-answer in the context of the entire reasoning chain.

For each SubQ-SubA pair, assess:
1. **Relevance**: Does the sub-answer directly address its sub-question?
2. **Correctness**: Is the sub-answer factually accurate?
3. **Completeness**: Does the sub-answer provide sufficient information?
4. **Consistency**: Is the sub-answer consistent with other sub-answers in the chain?
5. **Contribution**: Given the final answer, does this sub-answer contribute correctly?

Provide your analysis in the following format:

REASONING:
For each SubQ-SubA pair (1, 2, 3, ...), analyze its correctness considering the entire chain.

VERDICTS:
Provide one verdict per line in the exact format:
SubQ 1: [true/false]
SubQ 2: [true/false]
SubQ 3: [true/false]
...

Where "true" means the sub-answer is correct and "false" means it is incorrect or problematic."""

        return system_prompt, user_prompt
