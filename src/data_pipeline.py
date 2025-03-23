import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Ensure necessary NLTK datasets are downloaded
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    """Cleans and tokenizes the input text."""
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Load the search logs into a DataFrame
df = pd.read_csv("data/synthetic_search_logs.csv")

# Apply text preprocessing to search queries
df['query'] = df['query'].apply(preprocess_text)

# Generate search ranking metrics
df['ctr'] = df['click'] / df.groupby('query')['click'].transform('sum')  # Click-Through Rate (CTR)
df['adjusted_relevance'] = df['relevance'] * df['dwell_time']  # Weighted relevance score

# Save the processed dataset
df.to_csv("data/processed_search_logs.csv", index=False)

print("Data processing complete. Processed dataset saved to data/processed_search_logs.csv.")
