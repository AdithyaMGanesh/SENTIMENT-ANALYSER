import pandas as pd
from textblob import TextBlob

# Load the CSV
df = pd.read_csv("vaccination_tweets.csv")

# Function to assign sentiment
def get_sentiment(text):
    score = TextBlob(str(text)).sentiment.polarity
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

# Apply function
df["sentiment"] = df["text"].apply(get_sentiment)

# Preview result
print(df[["text", "sentiment"]].head())

# Save new file
df.to_csv("vaccination_tweets_labeled.csv", index=False)
print("✅ File saved as vaccination_tweets_labeled.csv")
