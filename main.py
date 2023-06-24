import streamlit as st
import nltk

nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def main():
    st.write("# Real-Time Sentiment Analysis")
    user_input = st.text_input("Please Enter Something : ")
    
    if user_input:
        s = SentimentIntensityAnalyzer()
        score = s.polarity_scores(user_input)

        if score["compound"] == 0:
            st.write("Neutral")
        elif score["compound"] > 0:
            st.write("Positive")
        else:
            st.write("Negative")

if __name__ == "__main__":
    main()
