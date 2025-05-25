import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import functional as F
import os

def run_image_detection(image_paths, output_dir):
    """
    Detect deepfake in images using FaceNet embeddings similarity.
    
    Args:
        image_paths (list of str): List of image file paths to analyze.
        output_dir (str): Directory to save annotated output images.
        
    Returns:
        dict: Mapping image filename -> dict with 'label' and 'confidence'
    """
    # Initialize face detector and embedding model
    mtcnn = MTCNN()
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

    threshold_face_similarity = 0.99  # Adjustable threshold

    previous_face_encoding = None
    results = {}

    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image {img_path}")
            results[os.path.basename(img_path)] = {"label": "Error", "confidence": 0.0}
            continue
        
        boxes, _ = mtcnn.detect(img)
        label = "No face detected"
        confidence = 0.0

        if boxes is not None and len(boxes) > 0:
            box = boxes[0].astype(int)
            face = img[box[1]:box[3], box[0]:box[2]]

            if face.size != 0:
                # Convert BGR to RGB before passing to FaceNet
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_rgb = cv2.resize(face_rgb, (160, 160))
                face_tensor = F.to_tensor(face_rgb).unsqueeze(0)
                current_face_encoding = facenet_model(face_tensor).detach().numpy().flatten()

                if previous_face_encoding is not None:
                    # Cosine similarity
                    similarity = np.dot(current_face_encoding, previous_face_encoding) / (
                        np.linalg.norm(current_face_encoding) * np.linalg.norm(previous_face_encoding)
                    )
                    confidence = similarity

                    if similarity < threshold_face_similarity:
                        label = "Deepfake"
                        color = (0, 0, 255)  # Red
                    else:
                        label = "Real"
                        color = (0, 255, 0)  # Green
                else:
                    label = "Reference"
                    confidence = 1.0
                    color = (255, 255, 0)  # Cyan

                previous_face_encoding = current_face_encoding

                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(img, f"{label} ({confidence:.2f})", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            else:
                label = "No valid face region"
        else:
            label = "No face detected"

        results[os.path.basename(img_path)] = {"label": label, "confidence": confidence}
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img)

    return results
