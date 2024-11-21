import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Ensure necessary resources are downloaded
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Combine datasets
tweets = positive_tweets + negative_tweets
labels = ['positive'] * len(positive_tweets) + ['negative'] * len(negative_tweets)

# Data cleaning function
def clean_tweet(tweet):
    tokens = word_tokenize(tweet)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    cleaned_tokens = [ps.stem(word) for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(cleaned_tokens)

# Clean all tweets
cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cleaned_tweets, labels, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output the results
print(f'Accuracy: {accuracy}')
print(report)
