"""Human-like evaluation using LLMs."""

import os
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

class ChatEvaluation(BaseModel):
    """Schema for chat evaluation metrics."""
    Helpfulness: int
    Fluency: int
    Appropriateness: int

class HumanEvaluator:
    def __init__(self):
        """Initialize the human evaluator with Groq models."""
        self.api_keys = os.getenv("GROQ_API_KEYS", "").split(",")
        self.models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "deepseek-r1-distill-llama-70b",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ]
        
        if len(self.api_keys) == 0:
            raise ValueError("No Groq API keys found in environment variables")
        
        self.current_model_index = 0
        self.setup_agent()

    def setup_agent(self):
        """Setup the Groq agent with the current model."""
        self.agent = Agent[None, ChatEvaluation](
            model=GroqModel(
                self.models[self.current_model_index],
                provider=GroqProvider(api_key=self.api_keys[0])
            ),
            system_prompt="""
            You are evaluating a chatbot's reply to a customer.
            
            Please rate the response on the following from 1 (poor) to 5 (excellent).
            Return only a JSON object with the fields:
            - Helpfulness: How well the response addresses the user's needs
            - Fluency: How natural and well-written the response is
            - Appropriateness: How suitable the tone and content are for customer service
            """,
            result_type=ChatEvaluation,
        )

    async def evaluate_single(self, query: str, response: str, max_retries: int = 3) -> Dict[str, int]:
        """Evaluate a single response with retries.
        
        Args:
            query: The user's query
            response: The generated response
            max_retries: Maximum number of retry attempts
        
        Returns:
            Dictionary containing evaluation scores
        """
        prompt = f"Customer Query:\n{query}\n\nChatbot Response:\n{response}"
        
        for attempt in range(max_retries):
            try:
                result = await self.agent.run(prompt)
                return {
                    'helpfulness': result.output.Helpfulness,
                    'fluency': result.output.Fluency,
                    'appropriateness': result.output.Appropriateness
                }
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Try switching to next model
                    self.current_model_index = (self.current_model_index + 1) % len(self.models)
                    self.setup_agent()
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return {
            'helpfulness': 0,
            'fluency': 0,
            'appropriateness': 0
        }

    async def evaluate_batch(self, queries: List[str], responses: List[str], batch_size: int = 5) -> Dict[str, float]:
        """Evaluate a batch of responses.
        
        Args:
            queries: List of user queries
            responses: List of generated responses
            batch_size: Number of evaluations to run in parallel
        
        Returns:
            Dictionary containing average scores
        """
        all_scores = []
        
        # Process in batches
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]
            
            # Run evaluations in parallel
            tasks = [
                self.evaluate_single(q, r)
                for q, r in zip(batch_queries, batch_responses)
            ]
            batch_scores = await asyncio.gather(*tasks)
            all_scores.extend(batch_scores)
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        # Calculate averages
        avg_scores = {
            'human_helpfulness': sum(s['helpfulness'] for s in all_scores) / len(all_scores),
            'human_fluency': sum(s['fluency'] for s in all_scores) / len(all_scores),
            'human_appropriateness': sum(s['appropriateness'] for s in all_scores) / len(all_scores)
        }
        avg_scores['human_overall'] = sum(avg_scores.values()) / 3
        
        return avg_scores 