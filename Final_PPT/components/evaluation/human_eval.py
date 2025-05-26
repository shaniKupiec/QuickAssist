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
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize the evaluator with a single Groq API key and a single model name.
        """
        if not api_key or not model_name:
            raise ValueError("Both api_key and model_name must be provided")
        self.api_key = api_key
        self.model_name = model_name
        self.setup_agent()

    def setup_agent(self):
        self.agent = Agent[None, ChatEvaluation](
            model=GroqModel(
                self.model_name,
                provider=GroqProvider(api_key=self.api_key)
            ),
            system_prompt=(
                "You are evaluating a chatbot's reply to a customer.\n\n"
                "Please rate the response on the following from 1 (poor) to 5 (excellent).\n"
                "Return only a valid JSON object, with fields:\n"
                "- Helpfulness: How well the response addresses the user's needs\n"
                "- Fluency: How natural and well-written the response is\n"
                "- Appropriateness: How suitable the tone and content are for customer service"
            ),
            result_type=ChatEvaluation,
        )

    async def evaluate_single(self, query: str, response: str, max_retries: int = 3) -> Dict[str, int]:
        prompt = f"Customer Query:\n{query}\n\nChatbot Response:\n{response}"
        for attempt in range(max_retries):
            try:
                result = await self.agent.run(prompt)
                return {
                    'Helpfulness': result.output.Helpfulness,
                    'Fluency': result.output.Fluency,
                    'Appropriateness': result.output.Appropriateness
                }
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(2 ** attempt)
        # If all attempts fail, return zeros
        return {'Helpfulness': 0, 'Fluency': 0, 'Appropriateness': 0}

    async def evaluate_batch(self, queries: List[str], responses: List[str], batch_size: int = 5) -> Dict[str, float]:
        assert len(queries) == len(responses), "Query and response lists must have same length"
        all_scores = []
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]
            tasks = [self.evaluate_single(q, r) for q, r in zip(batch_queries, batch_responses)]
            batch_scores = await asyncio.gather(*tasks)
            all_scores.extend(batch_scores)
            await asyncio.sleep(1)
        if not all_scores:
            return {k: 0 for k in ['human_helpfulness', 'human_fluency', 'human_appropriateness', 'human_overall']}
        avg_scores = {
            'human_helpfulness': sum(s['Helpfulness'] for s in all_scores) / len(all_scores),
            'human_fluency': sum(s['Fluency'] for s in all_scores) / len(all_scores),
            'human_appropriateness': sum(s['Appropriateness'] for s in all_scores) / len(all_scores)
        }
        avg_scores['human_overall'] = sum(avg_scores.values()) / 3
        return avg_scores