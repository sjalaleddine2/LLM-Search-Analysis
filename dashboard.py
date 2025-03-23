import pandas as pd
import streamlit as st
import plotly.express as px

# Load evaluated search logs
df = pd.read_csv("data/evaluated_search_logs.csv")

st.title("AI-Powered Search Quality Dashboard")
st.markdown("### Analyze search relevance and ranking performance")

# Display dataset preview
st.subheader("Search Log Overview")
st.dataframe(df.head())
st.subheader("Query Relevance Score Distribution")
fig = px.histogram(df, x="ai_relevance", nbins=20, title="Distribution of AI-Assigned Relevance Scores")
st.plotly_chart(fig)

st.subheader("Normalized AI Relevance Score Distribution")
fig_norm = px.histogram(df, x="normalized_ai_relevance", nbins=20)
st.plotly_chart(fig_norm)

st.subheader("Weighted Relevance Score Distribution")
fig_weighted = px.histogram(df, x="weighted_relevance", nbins=20)
st.plotly_chart(fig_weighted)

def mean_reciprocal_rank(df, score_col):
    reciprocal_ranks = []
    for query in df["query"].unique():
        query_results = df[df["query"] == query].sort_values(score_col, ascending=False)
        ranks = query_results["rank"].values
        first_relevant = (ranks == 1).nonzero()[0]
        if first_relevant.size > 0:
            reciprocal_ranks.append(1 / (first_relevant[0] + 1))
    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)

original_mrr = mean_reciprocal_rank(df, score_col='ai_relevance')
fine_tuned_mrr = mean_reciprocal_rank(df, score_col='weighted_relevance')

st.subheader("Search Ranking Comparison")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Original MRR", value=f"{original_mrr:.4f}")
with col2:
    st.metric(label="Fine-Tuned MRR", value=f"{fine_tuned_mrr:.4f}")

mrr_data = pd.DataFrame({
    "Model": ["Original", "Fine-Tuned"],
    "MRR": [original_mrr, fine_tuned_mrr]
})

fig_mrr = px.bar(mrr_data, x="Model", y="MRR", text="MRR", title="Original vs. Fine-Tuned MRR")
st.plotly_chart(fig_mrr)