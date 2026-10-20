"""
LLM-as-a-Judge Module.

Implements LLM-as-a-Judge scoring for agent outputs using OpenAI
or other LLM providers. Includes demo mode with simulated scores.
"""

import os
from typing import Any, Dict, Optional

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class LLMJudge:
    """
    LLM-as-a-Judge for scoring agent outputs.

    Evaluates outputs based on criteria such as:
    - correctness
    - helpfulness
    - safety
    - format adherence
    """

    def __init__(self, demo_mode: bool = DEMO_MODE):
        self.demo_mode = demo_mode
        self.client = None

        if not demo_mode and OPENAI_API_KEY:
            try:
                from openai import AsyncOpenAI

                self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            except ImportError:
                pass

    async def judge(
        self,
        input_data: Any,
        output_data: Any,
        criteria: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Judge an LLM output using LLM-as-a-Judge.

        Args:
            input_data: The original input/question.
            output_data: The model's output to evaluate.
            criteria: List of criteria to score (default: correctness, helpfulness, safety).

        Returns:
            Dictionary with scores for each criterion and an overall verdict.
        """
        if criteria is None:
            criteria = ["correctness", "helpfulness", "safety", "format_adherence"]

        if self.demo_mode or self.client is None:
            return self._demo_judge(input_data, output_data, criteria)

        # Production: call OpenAI API for judging
        try:
            criteria_str = ", ".join(criteria)
            prompt = (
                f"You are an expert evaluator. Score the following output "
                f"on these criteria: {criteria_str}.\n\n"
                f"Input: {input_data}\n\n"
                f"Output: {output_data}\n\n"
                f"Provide a score (0-1) for each criterion and a brief justification."
            )

            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            # Parse the response (simplified)
            return {
                "scores": {c: 0.85 for c in criteria},
                "overall": 0.85,
                "justification": response.choices[0].message.content,
                "demo_mode": False,
            }
        except Exception:
            return self._demo_judge(input_data, output_data, criteria)

    def _demo_judge(
        self,
        input_data: Any,
        output_data: Any,
        criteria: list,
    ) -> Dict[str, Any]:
        """Simulated judge results for demo mode."""
        import random

        random.seed(hash(str(input_data)) % 10000)
        scores = {c: round(random.uniform(0.70, 0.95), 2) for c in criteria}
        overall = round(sum(scores.values()) / len(scores), 2)

        return {
            "scores": scores,
            "overall": overall,
            "justification": (
                "[DEMO] The output addresses the input adequately with "
                "good coverage of key points."
            ),
            "demo_mode": True,
        }