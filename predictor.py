import neattext.functions as nfx
import os
from model_trainer import load_model
from faster_whisper import WhisperModel

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = nfx.remove_userhandles(text)
    text = nfx.remove_stopwords(text)
    text = nfx.remove_special_characters(text)
    text = nfx.remove_punctuations(text)
    return text.strip()

def predict_emotion_from_text(text):
    model, vectorizer, label_encoder = load_model()
    if not all([model, vectorizer, label_encoder]):
        print("Model not loaded. Train the model first.")
        return None
    text = clean_text(text)
    if not text:
        print("Nothing left after cleaning.")
        return None
    try:
        X = vectorizer.transform([text])
        pred = model.predict(X)
        emotion = label_encoder.inverse_transform(pred)[0]
        print(f"Predicted emotion: {emotion}")
        return emotion
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def predict_emotion_from_audio(audio_path, whisper_model_type="tiny.en"):
    if not os.path.exists(audio_path):
        print("Audio file not found.")
        return None
    try:
        model = WhisperModel(whisper_model_type, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path)
        text = " ".join([segment.text for segment in segments])
        print(f"Transcribed: {text}")
    except Exception as e:
        print(f"Faster Whisper error: {e}")
        return None
    return predict_emotion_from_text(text)
