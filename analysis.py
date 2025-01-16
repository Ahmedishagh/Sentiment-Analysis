import pandas as pd
import re
from transformers import pipeline
import matplotlib.pyplot as plt
# Load the dataset
dataframe = pd.read_csv("tweets.csv")

# Display the first few rows
print(dataframe.head())
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply preprocessing to the 'text' column
dataframe['text'] = dataframe['text'].apply(preprocess_text)
print(dataframe['text'].head())

# Filter French tweets
french_tweets = dataframe[dataframe['lang'] == 'fr']['text'].tolist()

# Display the first few French tweets
print("French Tweets:", french_tweets[:5])


# Load the sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis", model="./models/bert-sentiment")

# Test the model on the first French tweet
print(french_tweets[0])
print(sentiment_model(french_tweets[0]))
# Analyze sentiments for all French tweets
sentiments = [sentiment_model(tweet) for tweet in french_tweets]

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'text': french_tweets,
    'label': [sent[0]['label'] for sent in sentiments],
    'score': [sent[0]['score'] for sent in sentiments]
})

# Display the results
print(results_df.head())
# Export results to a CSV file
results_df.to_csv("sentiment_results.csv", index=False)
print("Results saved to sentiment_results.csv")


# Count each sentiment label
sentiment_counts = results_df['label'].value_counts()

# Create a bar chart
sentiment_counts.plot(kind='bar', color=['green', 'orange', 'red'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

