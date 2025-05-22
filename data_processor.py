import pandas as pd
import neattext.functions as nfx
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def load_data(csv_path='tweet_emotion.csv'):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None, None, None, None, None
    df = pd.read_csv(csv_path)
    df['Clean_Text'] = df['Text'].apply(lambda x: nfx.remove_userhandles(str(x)))
    df['Clean_Text'] = df['Clean_Text'].apply(lambda x: nfx.remove_stopwords(str(x)))
    df['Clean_Text'] = df['Clean_Text'].apply(lambda x: nfx.remove_special_characters(str(x)))
    df['Clean_Text'] = df['Clean_Text'].apply(lambda x: nfx.remove_punctuations(str(x)))
    label_encoder = LabelEncoder()
    df['Emotion_Encoded'] = label_encoder.fit_transform(df['Emotion'])
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(df['Clean_Text'])
    y = df['Emotion_Encoded']
    return X, y, label_encoder, tfidf_vectorizer, df
