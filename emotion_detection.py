import os
from data_processor import load_data
from model_trainer import train_model, save_model, load_model
from predictor import predict_emotion_from_text, predict_emotion_from_audio

DATA_FILE = 'tweet_emotion.csv'
AUDIO_FILE = 'sample_audio.wav'

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Missing: {DATA_FILE}")
        return
    model, vectorizer, label_encoder = load_model()
    if not all([model, vectorizer, label_encoder]):
        X, y, label_encoder, vectorizer, _ = load_data(DATA_FILE)
        if X is None:
            print("Data load failed.")
            return
        model = train_model(X, y)
        save_model(model, vectorizer, label_encoder)
    if os.path.exists(AUDIO_FILE):
        predict_emotion_from_audio(AUDIO_FILE)
    while True:
        text = input("Type a sentence (or 'quit'): ")
        if text.lower() == 'quit':
            break
        predict_emotion_from_text(text)

if __name__ == "__main__":
    main()
