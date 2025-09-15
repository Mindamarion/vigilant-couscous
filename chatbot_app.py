# chatbot_app.py
import streamlit as st
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')


nltk.download("punkt_tab")   # 👈 new requirement

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# -----------------------------
# Step 1: Preprocess the text
# -----------------------------
def preprocess(text_file):
    
    with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    # Split text into sentences
    sentences = sent_tokenize(raw_text)

    # Clean sentences
    stop_words = set(stopwords.words("english"))
    cleaned_sentences = []
    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        cleaned_sentences.append(" ".join(tokens))

    return sentences, cleaned_sentences  # keep both original and cleaned versions


# -----------------------------
# Step 2: Similarity function
# -----------------------------
def get_most_relevant_sentence(query, sentences, cleaned_sentences):
    # Add query to the corpus
    all_sentences = cleaned_sentences + [query]

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(all_sentences)

    # Compute cosine similarity
    similarity_scores = cosine_similarity(tfidf[-1], tfidf[:-1])

    # Get index of best match
    idx = similarity_scores.argsort()[0][-1]

    return sentences[idx]  # return original sentence (not cleaned)


# -----------------------------
# Step 3: Chatbot function
# -----------------------------
def chatbot(query, sentences, cleaned_sentences):
    if query.lower() in ["quit", "exit", "bye"]:
        return "Goodbye! 👋"
    response = get_most_relevant_sentence(query, sentences, cleaned_sentences)
    return response


# -----------------------------
# Step 4: Streamlit App
# -----------------------------
def main():
    st.title("📚 Text-based Chatbot")
    st.write("Ask me anything based on the knowledge in my text file!")

    # Load your chosen text file
    text_file =(r'C:\Users\LE\OneDrive\Desktop\chatbot\finding touth .txt')
    sentences, cleaned_sentences = preprocess(text_file)

    # User input
    user_query = st.text_input("Enter your question:")

    if user_query:
        answer = chatbot(user_query, sentences, cleaned_sentences)
        st.success(answer)


if __name__ == "__main__":
    main()
