import asyncio
from typing import List, Dict
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider


class ChatEvaluation(BaseModel):
    Helpfulness: int
    Fluency: int
    Appropriateness: int


class HumanEvaluator:
    def __init__(self, api_key: str, model_name: str, max_samples: int = 100):
        if not api_key or not model_name:
            raise ValueError("Both api_key and model_name must be provided")
        self.api_key = api_key
        self.model_name = model_name
        self.max_samples = max_samples
        self.evaluation_stopped = False
        self._setup_agent()

    def _setup_agent(self):
        self.agent = Agent[None, ChatEvaluation](
            model=GroqModel(self.model_name, provider=GroqProvider(api_key=self.api_key)),
            system_prompt=(
                "You are evaluating a chatbot's reply to a customer.\n\n"
                "Please rate the response on the following from 1 (poor) to 5 (excellent).\n"
                "Return only a valid JSON object, with fields:\n"
                "- Helpfulness\n- Fluency\n- Appropriateness"
            ),
            result_type=ChatEvaluation,
        )

    def _create_prompt(self, query: str, response: str) -> str:
        return f"Customer Query:\n{query}\n\nChatbot Response:\n{response}"

    async def _evaluate_single(self, index: int, query: str, response: str, max_retries: int = 6, base_delay: int = 2) -> Dict[str, int] | None:
        prompt = self._create_prompt(query, response)

        for attempt in range(max_retries):
            try:
                print(f"Evaluating {index + 1}/{self.max_samples}... (Attempt {attempt + 1})")
                result = await self.agent.run(prompt)
                return {
                    "Helpfulness": result.output.Helpfulness,
                    "Fluency": result.output.Fluency,
                    "Appropriateness": result.output.Appropriateness,
                    "average_score": (
                        result.output.Helpfulness +
                        result.output.Fluency +
                        result.output.Appropriateness
                    ) / 3
                }
            except Exception as e:
                delay = base_delay * (2 ** attempt)
                print(f"⚠️ Error evaluating sample {index + 1}: {str(e)} — retrying in {delay}s...")
                await asyncio.sleep(delay)

        print(f"🚫 Skipping sample {index + 1} after {max_retries} failed attempts.")
        return None

    async def evaluate_batch(self, queries: List[str], generated_responses: List[str]) -> List[Dict[str, int] | None]:
        results = []
        for i, (query, response) in enumerate(zip(queries, generated_responses)):
            if i >= self.max_samples:
                break
            if self.evaluation_stopped:
                print("🛑 Evaluation stopped early due to error.")
                break

            result = await self._evaluate_single(i, query, response)
            results.append(result)
            await asyncio.sleep(1.5)  # Rate limit buffer

        return results
    
    def avgMetricsHumanScore(self, evaluation_results: List[Dict[str, int] | None]) -> Dict[str, float]:
        """Computes average human evaluation metrics from evaluate_batch() results."""
        valid_scores = [s for s in evaluation_results if s is not None]

        if not valid_scores:
            print("⚠️ No valid human evaluation results.")
            return {}

        avg_helpfulness = sum(d['Helpfulness'] for d in valid_scores) / len(valid_scores)
        avg_fluency = sum(d['Fluency'] for d in valid_scores) / len(valid_scores)
        avg_appropriateness = sum(d['Appropriateness'] for d in valid_scores) / len(valid_scores)
        avg_total = sum(d['average_score'] for d in valid_scores) / len(valid_scores)

        return {
            "Human_Helpfulness": avg_helpfulness,
            "Human_Fluency": avg_fluency,
            "Human_Appropriateness": avg_appropriateness,
            "Human_Average": avg_total,
        }