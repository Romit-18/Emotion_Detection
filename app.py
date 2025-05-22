from flask import Flask, render_template, request, redirect, url_for, flash
import os
import uuid
from werkzeug.utils import secure_filename
from predictor import predict_emotion_from_text, predict_emotion_from_audio

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Utility Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file):
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    file.save(filepath)
    return filepath, filename

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    emotion_result = None
    user_text = None
    audio_file_name = None

    if request.method == 'POST':
        # Handle text input
        user_text = request.form.get('text_input', '').strip()
        if user_text:
            emotion_result = predict_emotion_from_text(user_text)
        
        # Handle audio upload
        elif 'audio_file' in request.files:
            file = request.files['audio_file']
            if file and allowed_file(file.filename):
                try:
                    filepath, audio_file_name = save_file(file)
                    emotion_result = predict_emotion_from_audio(filepath)
                except Exception as e:
                    flash(f"Error processing audio: {str(e)}", "error")
            else:
                flash("Invalid audio file format.", "warning")

    return render_template(
        'index.html',
        emotion=emotion_result,
        user_text=user_text,
        audio_file=audio_file_name
    )

# Main Execution
if __name__ == '__main__':
    app.run(debug=True)
