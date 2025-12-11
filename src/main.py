"""Main orchestrator for the Multi-Agent Debate RAG system."""
import json
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.retriever import FAISSRetriever
from src.rag.pdf_ingest import ingest_pdfs
from src.agents.base_debater import RagDebater
from src.agents.judge import QuizJudge
from src.utils.config import MAX_DEBATE_ROUNDS, VECTOR_STORE_DIR
from src.utils.logger import setup_logger

logger = setup_logger()

class MultiAgentDebateSystem:
    """Orchestrates the multi-agent debate and quiz generation."""
    
    def __init__(self, retriever: FAISSRetriever = None):
        """
        Initialize the multi-agent system.
        
        Args:
            retriever: Pre-built FAISSRetriever. If None, will try to load from disk.
        """
        # Load or create retriever
        if retriever is None:
            retriever = FAISSRetriever()
            try:
                retriever.load_index()
                logger.info("Loaded existing FAISS index")
            except FileNotFoundError:
                logger.warning("No existing index found. Please run pdf_ingest.py first.")
                raise
        
        self.retriever = retriever
        
        # Initialize agents
        self.proponent = RagDebater("TA_Proponent", "proponent", self.retriever)
        self.opponent = RagDebater("TA_Opponent", "opponent", self.retriever)
        self.judge = QuizJudge()
        
        logger.info("Multi-Agent Debate System initialized")
    
    def run_debate(self, topic: str, num_rounds: int = None) -> str:
        """
        Run the debate between Proponent and Opponent.
        
        Args:
            topic: The topic to debate
            num_rounds: Number of debate rounds (default from config)
            
        Returns:
            Full debate transcript
        """
        num_rounds = num_rounds or MAX_DEBATE_ROUNDS
        debate_history = []
        
        logger.info(f"Starting debate on topic: {topic} ({num_rounds} rounds)")
        
        for round_num in range(1, num_rounds + 1):
            logger.info(f"--- Round {round_num} ---")
            
            # Proponent speaks
            proponent_arg = self.proponent.think_and_speak(topic, debate_history)
            debate_history.append(f"Round {round_num} - Proponent: {proponent_arg}")
            logger.info(f"Proponent argument: {proponent_arg[:100]}...")
            
            # Opponent speaks
            opponent_arg = self.opponent.think_and_speak(topic, debate_history)
            debate_history.append(f"Round {round_num} - Opponent: {opponent_arg}")
            logger.info(f"Opponent argument: {opponent_arg[:100]}...")
        
        full_transcript = "\n".join(debate_history)
        logger.info("Debate completed")
        return full_transcript
    
    def generate_quiz(self, topic: str, debate_transcript: str = None) -> dict:
        """
        Generate quiz question from debate.
        
        Args:
            topic: The debate topic
            debate_transcript: Pre-existing transcript (if None, runs debate first)
            
        Returns:
            Quiz dictionary with question, answers, and explanation
        """
        if debate_transcript is None:
            debate_transcript = self.run_debate(topic)
        
        logger.info("Judge generating quiz from debate...")
        quiz = self.judge.generate_quiz(topic, debate_transcript)
        
        # Add metadata
        quiz["topic"] = topic
        quiz["debate_transcript"] = debate_transcript
        
        return quiz
    
    def save_quiz(self, quiz: dict, output_path: str):
        """Save quiz to JSON file."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(quiz, f, indent=2)
        logger.info(f"Quiz saved to {output_path}")

def main():
    """Main entry point for the system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Debate RAG System")
    parser.add_argument("--topic", type=str, required=True, help="Topic to debate and generate quiz for")
    parser.add_argument("--rounds", type=int, default=None, help="Number of debate rounds")
    parser.add_argument("--output", type=str, default="quiz_output.json", help="Output file path")
    parser.add_argument("--ingest", action="store_true", help="Ingest PDFs before running")
    
    args = parser.parse_args()
    
    # Ingest PDFs if requested
    if args.ingest:
        logger.info("Ingesting PDFs...")
        retriever = ingest_pdfs()
        if retriever is None:
            logger.error("PDF ingestion failed. Exiting.")
            return
    else:
        retriever = None
    
    # Initialize system
    try:
        system = MultiAgentDebateSystem(retriever)
    except FileNotFoundError:
        logger.error("No FAISS index found. Please run with --ingest first or run pdf_ingest.py separately.")
        return
    
    # Run debate and generate quiz
    quiz = system.generate_quiz(args.topic, None)
    
    # Print results
    print("\n" + "="*60)
    print("GENERATED QUIZ QUESTION")
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
    system.save_quiz(quiz, args.output)

if __name__ == "__main__":
    main()

