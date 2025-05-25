import os
import librosa
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from PIL import Image
import io
from moviepy.editor import VideoFileClip
from torchvision import transforms
from torch import nn

# Optional: Replace with your own trained audio model
class SimpleAudioClassifier(nn.Module):
    def __init__(self):
        super(SimpleAudioClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load model (replace with your fine-tuned version)
model = SimpleAudioClassifier()
model_path = "audio_model.pth"  # Update if needed
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def extract_audio_from_video(video_path, output_path="temp_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path, codec="pcm_s16le", verbose=False, logger=None)
    clip.close()
    return output_path

def audio_to_mel_spectrogram(wav_path, sr=16000, n_mels=128):
    y, sr = librosa.load(wav_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig = plt.figure(figsize=(2, 2), dpi=64)
    plt.axis('off')
    plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='magma')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert('L')
    image_tensor = transform(img).unsqueeze(0)

    # Convert image to base64 string
    import base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return image_tensor, img_base64


def classify_audio(video_path):
    try:
        audio_path = extract_audio_from_video(video_path)
        input_tensor, img_base64 = audio_to_mel_spectrogram(audio_path)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            confidence = prob[0][pred].item()

        label = "FAKE" if pred == 1 else "REAL"
        confidence_percent = int(confidence * 100)

        # Add base64 image header here before returning
        img_base64 = "data:image/png;base64," + img_base64

        return {
            "label": label,
            "confidence": confidence_percent,
            "spectrogram": img_base64
        }

    except Exception as e:
        return {"error": str(e)}