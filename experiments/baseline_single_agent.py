"""Baseline single-agent RAG system for comparison."""
import json
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import google.generativeai as genai
from src.rag.retriever import FAISSRetriever
from src.agents.prompts import get_baseline_prompt
from src.utils.config import GEMINI_API_KEY, GEMINI_MODEL
from src.utils.logger import setup_logger

logger = setup_logger()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

class BaselineSingleAgent:
    """Simple single-agent RAG for quiz generation (baseline)."""
    
    def __init__(self, retriever: FAISSRetriever):
        """Initialize baseline agent."""
        self.retriever = retriever
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        logger.info("Initialized BaselineSingleAgent")
    
    def generate_quiz(self, topic: str) -> dict:
        """
        Generate quiz directly from retrieved context (no debate).
        
        Args:
            topic: The topic to generate quiz for
            
        Returns:
            Quiz dictionary
        """
        # Retrieve relevant context
        try:
            retrieved_docs = self.retriever.retrieve(topic, k=5)
            context = "\n\n".join([f"[Context {i+1}]: {doc[0]}" for i, doc in enumerate(retrieved_docs)])
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            context = "No context retrieved."
        
        # Generate quiz
        prompt = get_baseline_prompt(context, topic)
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            quiz_data = json.loads(response_text)
            
            # Add metadata
            quiz_data["topic"] = topic
            quiz_data["method"] = "baseline_single_agent"
            
            logger.info("Successfully generated baseline quiz")
            return quiz_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {
                "question": "Error generating question.",
                "correct_answer": "N/A",
                "distractors": ["N/A", "N/A", "N/A"],
                "explanation": "JSON parsing failed.",
                "topic": topic,
                "method": "baseline_single_agent"
            }
        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            return {
                "question": f"Error: {str(e)}",
                "correct_answer": "N/A",
                "distractors": ["N/A", "N/A", "N/A"],
                "explanation": "Quiz generation failed.",
                "topic": topic,
                "method": "baseline_single_agent"
            }
    
    def save_quiz(self, quiz: dict, output_path: str):
        """Save quiz to JSON file."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(quiz, f, indent=2)
        logger.info(f"Quiz saved to {output_path}")

def main():
    """Main entry point for baseline system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Single-Agent RAG System")
    parser.add_argument("--topic", type=str, required=True, help="Topic to generate quiz for")
    parser.add_argument("--output", type=str, default="baseline_quiz_output.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Load retriever
    try:
        retriever = FAISSRetriever()
        retriever.load_index()
    except FileNotFoundError:
        logger.error("No FAISS index found. Please run pdf_ingest.py first.")
        return
    
    # Generate quiz
    agent = BaselineSingleAgent(retriever)
    quiz = agent.generate_quiz(args.topic)
    
    # Print results
    print("\n" + "="*60)
    print("BASELINE QUIZ QUESTION")
    print("="*60)
    print(f"\nTopic: {quiz['topic']}")
    print(f"\nQuestion: {quiz['question']}")
    print(f"\nCorrect Answer: {quiz['correct_answer']}")
    print(f"\nDistractors:")
    for i, distractor in enumerate(quiz['distractors'], 1):
        print(f"  {i}. {distractor}")
    print(f"\nExplanation: {quiz['explanation']}")
    print("="*60 + "\n")
    
    # Save to file
    agent.save_quiz(quiz, args.output)

if __name__ == "__main__":
    main()

