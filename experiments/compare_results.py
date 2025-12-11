"""Script to run batch comparison between baseline and multi-agent systems."""
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger()

def run_baseline(topic: str, output_dir: str = "experiments/results") -> str:
    """Run baseline system and return output file path."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"baseline_{topic.replace(' ', '_')}.json")
    
    logger.info(f"Running baseline for topic: {topic}")
    cmd = [
        "python", "experiments/baseline_single_agent.py",
        "--topic", topic,
        "--output", output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Baseline failed: {e}")
        return None

def run_multiagent(topic: str, rounds: int = 2, output_dir: str = "experiments/results") -> str:
    """Run multi-agent system and return output file path."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"multiagent_{topic.replace(' ', '_')}.json")
    
    logger.info(f"Running multi-agent for topic: {topic}")
    cmd = [
        "python", "src/main.py",
        "--topic", topic,
        "--rounds", str(rounds),
        "--output", output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Multi-agent failed: {e}")
        return None

def run_evaluation(baseline_file: str, multiagent_file: str, output_dir: str = "experiments/results") -> str:
    """Run evaluation script and return results file path."""
    results_file = os.path.join(output_dir, f"evaluation_{Path(baseline_file).stem.replace('baseline_', '')}.json")
    
    logger.info("Running evaluation...")
    cmd = [
        "python", "experiments/evaluate_distractors.py",
        "--baseline", baseline_file,
        "--multiagent", multiagent_file,
        "--output", results_file
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        return results_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e}")
        return None

def main():
    """Main entry point for batch comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run batch comparison between systems")
    parser.add_argument("--topics", type=str, nargs="+", required=True, 
                       help="List of topics to test (e.g., 'Machine Learning' 'Neural Networks')")
    parser.add_argument("--rounds", type=int, default=2, help="Number of debate rounds")
    parser.add_argument("--output-dir", type=str, default="experiments/results", 
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    all_results = []
    
    for topic in args.topics:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing topic: {topic}")
        logger.info(f"{'='*60}\n")
        
        # Run baseline
        baseline_file = run_baseline(topic, args.output_dir)
        if not baseline_file:
            logger.error(f"Skipping topic {topic} due to baseline failure")
            continue
        
        # Run multi-agent
        multiagent_file = run_multiagent(topic, args.rounds, args.output_dir)
        if not multiagent_file:
            logger.error(f"Skipping topic {topic} due to multi-agent failure")
            continue
        
        # Run evaluation
        eval_file = run_evaluation(baseline_file, multiagent_file, args.output_dir)
        if eval_file:
            with open(eval_file, 'r') as f:
                results = json.load(f)
                results['topic'] = topic
                all_results.append(results)
    
    # Aggregate results
    if all_results:
        summary = {
            'num_topics': len(all_results),
            'average_improvement': {
                'cosine_similarity': sum(r['improvement']['cosine_similarity'] for r in all_results) / len(all_results),
                'bertscore_f1': sum(r['improvement']['bertscore_f1'] for r in all_results) / len(all_results)
            },
            'detailed_results': all_results
        }
        
        summary_file = os.path.join(args.output_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Topics tested: {summary['num_topics']}")
        logger.info(f"Average Cosine Similarity Improvement: {summary['average_improvement']['cosine_similarity']:.3f}")
        logger.info(f"Average BERTScore F1 Improvement: {summary['average_improvement']['bertscore_f1']:.3f}")
        logger.info(f"Full summary saved to: {summary_file}")

if __name__ == "__main__":
    main()

