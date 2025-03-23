# LLM-Search-Analysis

This repository contains an AI-based pipeline for evaluating the relevance of search results using transformer models. The system processes search logs, applies model-based relevance scoring, incorporates user interaction signals, and provides interactive visualizations through a Streamlit dashboard.

## Overview
- Leverages TinyBERT for scoring query-result pairs
- Processes search data with CTR, dwell time, and relevance annotations
- Enhances ranking using behavior-weighted scoring
- Evaluates performance using Mean Reciprocal Rank (MRR)
- Includes a Streamlit dashboard for visualizing ranking quality
- All steps are integrated via a single `main.py` script

## Results
- **Baseline MRR:** 0.4020  
- **Behavior-Weighted MRR:** 0.5046  

The observed 25% improvement highlights the impact of incorporating behavioral signals into the ranking process.

## Running the Pipeline

To generate synthetic data before running the pipeline, execute the synthetic data script:
```bash
python src/generate_synthetic_data.py
```
This will create a sample search log in the `data/` directory.

Execute all components through the main script:
```bash
python main.py
```
Steps included:
1. Search log preprocessing
2. AI relevance scoring and metric evaluation
3. Optionally launching the dashboard for visualization

Upon prompt:
```bash
Launch Streamlit dashboard? (y/n): y
```
Access the dashboard at: `http://localhost:8501`

## Creating Synthetic Search Data

You can also manually create a CSV file if preferred, but the recommended method is to run the script above to generate a properly formatted example.
In the absence of real search data, you can use a CSV file with this structure:

Example: `data/synthetic_search_logs.csv`
```csv
query,result,rank,click,dwell_time,relevance
learn python,Python tutorial,1,0,0,0
learn python,Learn Python fast,2,1,35,1
learn python,Python for beginners,3,0,0,0
cheap flights tokyo,Flight deals to Tokyo,1,1,60,1
cheap flights tokyo,Cheapest Tokyo flights,2,1,25,1
cheap flights tokyo,Best airlines to Japan,3,0,0,0
```

Running `main.py` will automatically process this file if placed in the `data/` directory.

## Dashboard Features
- Histogram of AI relevance scores
- Comparison of original vs. fine-tuned MRR
- Preview of the processed search log

## System Overview
1. Load and preprocess log data
2. Score relevance using TinyBERT via Hugging Face
3. Normalize and weight scores using behavioral signals
4. Compute MRR for original and enhanced scores
5. Present results in an interactive dashboard

## Technologies Used
- Python (Pandas, NumPy)
- Hugging Face Transformers (TinyBERT)
- Streamlit & Plotly (for UI/visualization)

## Project Structure
```
LLM-Search-Analysis/
├── data/                         # Search log input/output
├── src/
│   ├── generate_synthetic_data.py  # Creates example synthetic search logs
│   ├── evaluate_queries.py         # Scoring and metric evaluation
│   └── data_pipeline.py            # Preprocessing pipeline
├── dashboard.py                  # Interactive dashboard
├── main.py                       # Entry point for the full workflow
├── README.md                     # Project summary and instructions
```


