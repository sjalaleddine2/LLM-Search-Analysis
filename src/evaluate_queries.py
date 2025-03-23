import pandas as pd
from transformers import pipeline
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load processed search logs
df = pd.read_csv("data/processed_search_logs.csv")

# Load the pre-trained model outside the function to avoid reloading every call
model = pipeline("text-classification", model="cross-encoder/ms-marco-TinyBERT-L-2-v2")

def evaluate_relevance(query, result):
    """Evaluates the relevance of a search result given a query using a Hugging Face model."""
    input_text = f"Query: {query} | Result: {result}"
    response = model(input_text)
    return response[0]['score']

def mean_reciprocal_rank(df, score_col):
    reciprocal_ranks = []
    for query in df["query"].unique():
        query_results = df[df["query"] == query].sort_values(score_col, ascending=False)
        ranks = query_results["rank"].values
        first_relevant = (ranks == 1).nonzero()[0]
        if first_relevant.size > 0:
            reciprocal_ranks.append(1 / (first_relevant[0] + 1))
    mrr_score = round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)
    
    return mrr_score

# Apply the model to each query-result pair
df["ai_relevance"] = df.apply(lambda row: evaluate_relevance(row["query"], row["result"]), axis=1)

# Normalize AI relevance scores
scaler = MinMaxScaler()
df["normalized_ai_relevance"] = scaler.fit_transform(df[["ai_relevance"]])

# Calculate and store MRRs
print(f"Original MRR: {mean_reciprocal_rank(df, score_col='ai_relevance'):.4f}")
print(f"Normalized MRR: {mean_reciprocal_rank(df, score_col='normalized_ai_relevance'):.4f}")

df['behavior_weight'] = (df['click'] + 1) * (df['dwell_time'] + 1)
df['weighted_relevance'] = df['normalized_ai_relevance'] * df['behavior_weight']

df['original_score'] = df['ai_relevance']
df['fine_tuned_score'] = df['weighted_relevance']

print(f"Fine-tuned MRR: {mean_reciprocal_rank(df, score_col='weighted_relevance'):.4f}")

# Save results
df.to_csv("data/evaluated_search_logs.csv", index=False)



