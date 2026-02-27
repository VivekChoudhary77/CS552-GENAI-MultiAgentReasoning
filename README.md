# CS552-GENAI-MultiAgentReasoning

A comparative study of adversarial multi-agent Retrieval-Augmented Generation (RAG) for generating high-quality educational assessments and multiple-choice distractors.

## Overview

This project implements a Multi-Agent RAG system that automates the creation of educational assessments through structured adversarial debate. Instead of a single agent generating questions from retrieved context, the system uses three specialized agents: a Proponent that retrieves and presents supporting evidence, an Opponent that retrieves and presents contradictory or limiting evidence, and a Judge that synthesizes the debate to produce multiple-choice questions with carefully designed distractors. The adversarial setup encourages broader coverage of the topic and reduces bias, leading to questions that test reasoning rather than simple recall.

## What Is Done

### System Architecture

The pipeline has three stages: document ingestion, adversarial debate, and quiz generation.

**Document processing and retrieval.** Academic PDFs are placed in `data/raw_pdfs/`. The ingestion script (`src/rag/pdf_ingest.py`) extracts text using `pypdf.PdfReader`, then splits it into chunks with `langchain.text_splitter.RecursiveCharacterTextSplitter` (default chunk size of 1000 characters and 200-character overlap, measured by `len`). Chunks are embedded with `sentence-transformers` (`all-MiniLM-L6-v2`) and indexed using `faiss.IndexFlatL2` for exact L2-distance search. The index and document metadata are saved to `data/vector_store/`. The retriever (`src/rag/retriever.py`) takes a text query, encodes it with the same model, and returns the top-k nearest chunks by L2 distance.

**Multi-agent debate.** The Proponent and Opponent agents (`src/agents/base_debater.py`) each first ask Google Gemini to generate a short search query from the current debate state. That query is run through the FAISS retriever to get relevant evidence. The agent then passes the topic, retrieved evidence, and prior debate history into a role-specific prompt (`src/agents/prompts.py`) and calls Gemini to produce an argument. Debate runs for a configurable number of rounds (default two, set via `MAX_DEBATE_ROUNDS` in `src/utils/config.py`).

**Quiz generation.** After all debate rounds, the Judge agent (`src/agents/judge.py`) receives the full transcript and calls Gemini with a prompt that instructs it to output a JSON object containing a question, the correct answer, three distractors (hard, medium, easy), and an explanation. The hard distractor is meant to use partial truths from the Opponent's arguments, the medium distractor targets common misconceptions from the debate, and the easy distractor is factually incorrect. The Judge parses the model's response as JSON and validates the required keys.

**Baseline.** A single-agent RAG baseline (`experiments/baseline_single_agent.py`) retrieves the top-k chunks for a topic and passes them directly to Gemini with a simpler prompt, generating a quiz question with no debate. This serves as the control for comparison.

**Evaluation.** Distractor quality is measured by two metrics in `experiments/evaluate_distractors.py`:
- Cosine similarity between the correct answer embedding and each distractor embedding, computed via `sklearn.metrics.pairwise.cosine_similarity` using the same `all-MiniLM-L6-v2` sentence embeddings.
- BERTScore (precision, recall, F1) between the correct answer and each distractor, computed via the `bert-score` library.

The script `experiments/compare_results.py` automates running both systems on a list of topics and aggregates evaluation results into a summary JSON.

### Project Structure

```
src/
  main.py               # Orchestrator: runs debate rounds and quiz generation
  rag/
    pdf_ingest.py        # PDF text extraction, chunking, FAISS index building
    retriever.py         # FAISSRetriever class (build, save, load, query)
  agents/
    base_debater.py      # RagDebater class (Proponent and Opponent roles)
    judge.py             # QuizJudge class (synthesizes debate into quiz JSON)
    prompts.py           # All system prompts and prompt-building functions
  utils/
    config.py            # Loads .env; defines model, chunking, and path settings
    logger.py            # Shared logging setup
experiments/
  baseline_single_agent.py   # Single-agent RAG baseline for comparison
  evaluate_distractors.py    # Cosine similarity and BERTScore evaluation
  compare_results.py         # Batch runner: baseline vs. multi-agent across topics
materials/
  project_report.tex         # Full project report with methodology and results
```

### Dependencies

All dependencies are listed in `requirements.txt`:

`langchain`, `faiss-cpu`, `google-generativeai`, `pypdf`, `python-dotenv`, `bert-score`, `pandas`, `numpy`, `tiktoken`, `langchain-community`, `langchain-google-genai`, `sentence-transformers`, `scikit-learn`

The core libraries actively used in the source code are `google-generativeai` (Gemini API calls), `sentence-transformers` (embedding), `faiss-cpu` (vector index), `pypdf` (PDF reading), `langchain` (text splitting), `scikit-learn` (cosine similarity in evaluation), `bert-score` (BERTScore in evaluation), `numpy` (array operations), and `python-dotenv` (environment variable loading).

## Setup and Usage

1. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root with at least:
   ```
   GEMINI_API_KEY=your_key_here
   ```
   Optional variables (with defaults shown) include `GEMINI_MODEL=gemini-pro`, `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`, `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`, `TOP_K_RETRIEVAL=3`, and `MAX_DEBATE_ROUNDS=2`.
3. Place PDF documents in `data/raw_pdfs/` and build the FAISS index:
   ```
   python src/rag/pdf_ingest.py
   ```
4. Run the multi-agent system:
   ```
   python src/main.py --topic "Your Topic" [--rounds N] [--output path] [--ingest]
   ```
5. Run the baseline:
   ```
   python experiments/baseline_single_agent.py --topic "Your Topic" [--output path]
   ```
6. Evaluate two quiz JSON files:
   ```
   python experiments/evaluate_distractors.py --baseline path --multiagent path [--output path]
   ```
7. Batch comparison across multiple topics:
   ```
   python experiments/compare_results.py --topics "Topic1" "Topic2" [--rounds N] [--output-dir dir]
   ```

## Results

Evaluation was conducted on three academic topics (Machine Learning, Neural Networks, Transformers). The following table is reproduced from the project report (`materials/project_report.tex`):

| Topic             | System      | Cosine Sim. | BERTScore F1 |
|-------------------|-------------|-------------|--------------|
| Machine Learning  | Baseline    | 0.342       | 0.867        |
| Machine Learning  | Multi-Agent | 0.277       | 0.860        |
| Neural Networks   | Baseline    | 0.392       | 0.880        |
| Neural Networks   | Multi-Agent | 1.000       | 1.000        |
| Transformers      | Baseline    | 1.000       | 1.000        |
| Transformers      | Multi-Agent | 1.000       | 1.000        |

Average improvement across topics: +18.1% in cosine similarity, +3.8% in BERTScore F1.

Key observations from the report:
- For Machine Learning, the multi-agent system's lower cosine similarity (0.277 vs. 0.342) indicates that its distractors are more semantically distinct from the correct answer while remaining plausible.
- For Neural Networks, the multi-agent system achieved perfect similarity scores, suggesting distractors that closely mirror the correct answer's semantic structure while remaining factually distinct.
- Qualitative analysis in the report shows that the multi-agent approach produces questions requiring deeper reasoning and distractors with appropriate difficulty gradation.

Full methodology, qualitative analysis, and discussion of future work are in `materials/project_report.tex`.

## Authors

Vivek Choudhary, Kumar Srinivas Bobba
