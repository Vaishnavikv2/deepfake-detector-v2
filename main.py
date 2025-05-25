from flask import Flask, jsonify, render_template, request, redirect, url_for
import os
import json
from datetime import datetime
from time import time as current_time
import importlib
from werkzeug.utils import secure_filename
from image_detector import run_image_detection
from audio_detector import classify_audio

app = Flask(__name__)

# Directories
app.config['UPLOAD_VIDEO_FOLDER'] = 'static/videos'
app.config['UPLOAD_IMAGE_FOLDER'] = 'static/images'
app.config['UPLOAD_AUDIO_FOLDER'] = 'static/audios'

# Ensure folders exist
for folder in [app.config['UPLOAD_VIDEO_FOLDER'], app.config['UPLOAD_IMAGE_FOLDER'], app.config['UPLOAD_AUDIO_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  # Buttons for audio, video, image

# Video Upload
@app.route('/upload-video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            timestamp = int(current_time())
            filename = f"video_{timestamp}.mp4"
            save_path = os.path.join(app.config['UPLOAD_VIDEO_FOLDER'], filename)
            file.save(save_path)

            video_path2 = os.path.join(app.config['UPLOAD_VIDEO_FOLDER'], f"1{filename}")
            module = importlib.import_module("deepfake_detector")
            result = getattr(module, "run")(save_path, video_path2)

            video_info = {
                'name': file.filename,
                'size': f"{os.path.getsize(save_path) / 1024:.2f} KB",
                'source': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'per': result
            }
            return render_template('video_result.html', video_url=video_path2, video_info=video_info)
        else:
            return "Unsupported video type", 400
    return render_template('upload_video_form.html')

# Image Upload
@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_IMAGE_FOLDER'], filename)
            file.save(save_path)

            results = run_image_detection([save_path], app.config['UPLOAD_IMAGE_FOLDER'])
            return render_template('image_result.html', image_url=save_path, detection=results[filename])
        else:
            return "Unsupported image type", 400
    return render_template('upload_image_form.html')

# Audio Upload
@app.route('/upload-audio', methods=['GET', 'POST'])
def upload_audio():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and file.filename.lower().endswith(('.wav', '.mp3', '.aac')):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_AUDIO_FOLDER'], filename)
            file.save(save_path)

            result = classify_audio(save_path)
            return render_template('audio_result.html', audio_url=save_path, detection=result)
        else:
            return "Unsupported audio type", 400
    return render_template('upload_audio_form.html')

if __name__ == '__main__':
    app.run(debug=True)
