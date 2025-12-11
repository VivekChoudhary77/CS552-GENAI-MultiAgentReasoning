"""Base debater class for Proponent and Opponent agents."""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import google.generativeai as genai
from typing import List, Optional
from src.utils.config import GEMINI_API_KEY, GEMINI_MODEL
from src.rag.retriever import FAISSRetriever
from src.agents.prompts import get_proponent_prompt, get_opponent_prompt
from src.utils.logger import setup_logger

logger = setup_logger()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

class RagDebater:
    """RAG-enhanced debater that retrieves evidence before arguing."""
    
    def __init__(self, name: str, role: str, retriever: FAISSRetriever):
        """
        Initialize a RAG debater.
        
        Args:
            name: Agent name (e.g., "TA_Proponent")
            role: "proponent" or "opponent"
            retriever: FAISSRetriever instance
        """
        self.name = name
        self.role = role.lower()
        if self.role not in ["proponent", "opponent"]:
            raise ValueError(f"Role must be 'proponent' or 'opponent', got {role}")
        
        self.retriever = retriever
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.debate_history = []
        
        logger.info(f"Initialized {self.name} as {self.role}")
    
    def _generate_search_query(self, topic: str, debate_history: List[str]) -> str:
        """Generate a search query based on topic and debate history."""
        if self.role == "proponent":
            query_prompt = f"""You are a Proponent researching the topic: "{topic}"

Based on the debate history, what specific information should you search for to support your argument?

Debate History:
{chr(10).join(debate_history[-2:]) if debate_history else "No previous arguments."}

Generate a concise search query (1-5 words) to find supporting evidence:"""
        else:  # opponent
            query_prompt = f"""You are an Opponent researching the topic: "{topic}"

Based on the debate history, what specific information should you search for to challenge or contradict the argument?

Debate History:
{chr(10).join(debate_history[-2:]) if debate_history else "No previous arguments."}

Generate a concise search query (1-5 words) to find contradictory evidence:"""
        
        try:
            response = self.model.generate_content(query_prompt)
            search_query = response.text.strip()
            logger.debug(f"{self.name} generated search query: {search_query}")
            return search_query
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            # Fallback to topic-based query
            return topic if self.role == "proponent" else f"limitations of {topic}"
    
    def think_and_speak(self, topic: str, debate_history: List[str] = None) -> str:
        """
        Retrieve evidence and generate an argument.
        
        Args:
            topic: The debate topic
            debate_history: Previous arguments in the debate
            
        Returns:
            The agent's argument
        """
        debate_history = debate_history or []
        
        # Generate search query
        search_query = self._generate_search_query(topic, debate_history)
        
        # Retrieve evidence
        try:
            retrieved_docs = self.retriever.retrieve(search_query, k=3)
            evidence = "\n\n".join([f"[Evidence {i+1}]: {doc[0]}" for i, doc in enumerate(retrieved_docs)])
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            evidence = "No evidence retrieved."
        
        # Generate argument based on role
        if self.role == "proponent":
            prompt = get_proponent_prompt(topic, evidence, debate_history)
        else:
            prompt = get_opponent_prompt(topic, evidence, debate_history)
        
        try:
            response = self.model.generate_content(prompt)
            argument = response.text.strip()
            self.debate_history.append(f"{self.name}: {argument}")
            logger.info(f"{self.name} generated argument ({len(argument)} chars)")
            return argument
        except Exception as e:
            logger.error(f"Error generating argument for {self.name}: {e}")
            return f"{self.name} encountered an error: {str(e)}"

