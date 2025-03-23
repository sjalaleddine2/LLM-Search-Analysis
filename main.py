# main.py

"""
Main runner script to execute the full AI-powered search evaluation pipeline:
1. Data preprocessing
2. Relevance scoring and evaluation
3. Optional dashboard launch
"""

import subprocess
import os
import stat

# Step 1: Preprocess the search log data
print("[Step 1] Running data preprocessing pipeline...")
subprocess.run(["python", "src/data_pipeline.py"], check=True)

# Step 2: Run AI-based evaluation (relevance scoring and MRR)
print("[Step 2] Evaluating query-result relevance...")
subprocess.run(["python", "src/evaluate_queries.py"], check=True)

# Step 3: (Optional) Launch Streamlit dashboard
launch_dashboard = input("\nLaunch Streamlit dashboard? (y/n): ").strip().lower()
if launch_dashboard == 'y':
    print("[Step 3] Launching dashboard at http://localhost:8501...")
    subprocess.run(["streamlit", "run", "dashboard.py"])
else:
    print("[Done] Pipeline complete. Dashboard not launched.")
