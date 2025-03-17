import random
import pandas as pd

# Sample search queries
queries = [
    "best laptop for gaming", "how to learn Python", "weather in New York",
    "best programming languages 2025", "cheap flights to Tokyo"
]

# Sample results with relevance scores (0 - Irrelevant, 1 - Relevant)
search_results = [
    ["Laptop A review", "Laptop B specs", "Top gaming laptops"],
    ["Python tutorial", "Learn Python fast", "Python for beginners"],
    ["NYC weather today", "Weather forecast", "New York climate"],
    ["Best languages to learn", "Top programming languages", "Most in-demand coding skills"],
    ["Flight deals to Tokyo", "Cheapest Tokyo flights", "Best airlines to Japan"]
]

# Generate synthetic logs
data = []
for i in range(1000):  # 1000 synthetic searches
    q_idx = random.randint(0, len(queries) - 1)
    query = queries[q_idx]
    results = search_results[q_idx]

    for rank, result in enumerate(results):
        click = random.choice([0, 1])  # Simulating if the user clicked
        dwell_time = random.randint(2, 60) if click else 0  # Time spent on page
        relevance = 1 if click else random.choice([0, 1])  # Randomized relevance score

        data.append([query, result, rank+1, click, dwell_time, relevance])

# Create DataFrame
df = pd.DataFrame(data, columns=["query", "result", "rank", "click", "dwell_time", "relevance"])

# Save to CSV
df.to_csv("../data/synthetic_search_logs.csv", index=False)
print("Synthetic search logs generated and saved!")
