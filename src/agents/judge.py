"""Judge agent that synthesizes debate and generates quiz questions."""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import google.generativeai as genai
from typing import Dict, Optional
from src.utils.config import GEMINI_API_KEY, GEMINI_MODEL
from src.agents.prompts import get_judge_prompt
from src.utils.logger import setup_logger

logger = setup_logger()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

class QuizJudge:
    """Judge agent that generates quiz questions from debate transcript."""
    
    def __init__(self):
        """Initialize the Judge agent."""
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        logger.info("Initialized QuizJudge")
    
    def generate_quiz(self, topic: str, full_debate_transcript: str) -> Dict:
        """
        Generate a quiz question from the debate transcript.
        
        Args:
            topic: The debate topic
            full_debate_transcript: Complete debate history
            
        Returns:
            Dictionary with question, correct_answer, distractors, and explanation
        """
        prompt = get_judge_prompt(topic, full_debate_transcript)
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from response (in case it's wrapped in markdown)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            quiz_data = json.loads(response_text)
            
            # Validate structure
            required_keys = ["question", "correct_answer", "distractors", "explanation"]
            for key in required_keys:
                if key not in quiz_data:
                    raise ValueError(f"Missing required key: {key}")
            
            if not isinstance(quiz_data["distractors"], list) or len(quiz_data["distractors"]) != 3:
                raise ValueError("distractors must be a list of 3 items")
            
            logger.info("Successfully generated quiz question")
            return quiz_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            # Return a fallback structure
            return {
                "question": "Error generating question. Please check logs.",
                "correct_answer": "N/A",
                "distractors": ["N/A", "N/A", "N/A"],
                "explanation": "JSON parsing failed."
            }
        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            return {
                "question": f"Error: {str(e)}",
                "correct_answer": "N/A",
                "distractors": ["N/A", "N/A", "N/A"],
                "explanation": "Quiz generation failed."
            }

