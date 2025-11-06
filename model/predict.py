#predict.py
import argparse
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import os

def main():
    parser = argparse.ArgumentParser(
        description="Classify an image as 'artificial' or 'human' using the SDXL-Deepfake-Detector."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file")
    args = parser.parse_args()

    # Validate image path
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    # Load model and feature extractor from Hugging Face Hub
    model_name = "./SDXL-Deepfake-Detector"
    print(f"Loading model '{model_name}'...")
    model = AutoModelForImageClassification.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Running on device: {device}")

    # Load and preprocess image
    image = Image.open(args.image).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]

    # Output
    print(f"Prediction Result")
    print(f"Class Index: {predicted_class_idx}")
    print(f"Label      : {predicted_label}")

if __name__ == "__main__":
    main()
