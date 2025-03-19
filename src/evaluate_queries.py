import pandas as pd
from transformers import pipeline
import numpy as np

# Load processed search logs
df = pd.read_csv("../data/processed_search_logs.csv")

# Load the pre-trained model outside the function to avoid reloading every call
model = pipeline("text-classification", model="cross-encoder/ms-marco-TinyBERT-L-2-v2")

def evaluate_relevance(query, result):
    """Evaluates the relevance of a search result given a query using a Hugging Face model."""
    input_text = f"Query: {query} | Result: {result}"
    response = model(input_text)
    return response[0]['score']

def mean_reciprocal_rank(df):
    reciprocal_ranks = []
    for query in df["query"].unique():
        query_results = df[df["query"] == query].sort_values("ai_relevance", ascending=False)
        ranks = query_results["rank"].values
        first_relevant = np.where(ranks == 1)[0]
        if first_relevant.size > 0:
            reciprocal_ranks.append(1 / (first_relevant[0] + 1))
    mrr_score = np.mean(reciprocal_ranks)
    
    # Store result in a CSV file
    with open("../data/evaluation_scores.csv", "a") as f:
        f.write(f"MRR,{mrr_score:.4f}\n")
    
    return mrr_score

# Apply the model to each query-result pair
df["ai_relevance"] = df.apply(lambda row: evaluate_relevance(row["query"], row["result"]), axis=1)

# Save results
df.to_csv("../data/evaluated_search_logs.csv", index=False)

# Calculate and store MRR
print(f"Mean Reciprocal Rank (MRR): {mean_reciprocal_rank(df):.4f}")
