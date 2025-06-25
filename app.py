from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Load saved model, vectorizer, and label encoder
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# NLP tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text cleaning function (same as training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = text.strip().split()
    return " ".join([stemmer.stem(word) for word in tokens if word not in stop_words])

# Route: homepage with form
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        tweet = request.form["tweet"]
        cleaned = clean_text(tweet)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)
        sentiment = label_encoder.inverse_transform(prediction)[0]
    return render_template("index.html", sentiment=sentiment)

# Start the app
if __name__ == "__main__":
    app.run(debug=True)
