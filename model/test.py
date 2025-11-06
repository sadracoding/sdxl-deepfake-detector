# evaluate.py
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def main():
    # evaluate.py
    test_dir = Path("./dataset/test")
    model_dir = "./SDXL-Deepfake-Detector" #Or just SDXL-Deepfake-Detector if you want to use huggingface
    batch_size = 8

    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Map dataset folders to model labels
    folder_to_model_label = {"fake": "artificial", "real": "human"}
    for folder in folder_to_model_label:
        if not (test_dir / folder).is_dir():
            raise FileNotFoundError(f"Folder '{folder}' missing in {test_dir}")

    # Load model and get label mapping
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    label2id = {v: k for k, v in model.config.id2label.items()}

    # Validate model labels
    for lbl in folder_to_model_label.values():
        if lbl not in label2id:
            raise ValueError(f"Model missing label '{lbl}'. Found: {list(label2id.keys())}")

    folder_to_label_id = {f: label2id[lbl] for f, lbl in folder_to_model_label.items()}

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Collect images and true labels
    all_image_paths = []
    all_true_labels = []
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    for folder, label_id in folder_to_label_id.items():
        folder_path = test_dir / folder
        image_paths = [p for p in folder_path.rglob("*") if p.suffix.lower() in valid_ext and p.is_file()]
        all_image_paths.extend(image_paths)
        all_true_labels.extend([label_id] * len(image_paths))

    if not all_image_paths:
        raise ValueError("No valid images found in test directory.")

    # Inference
    all_preds = []
    for i in range(0, len(all_image_paths), batch_size):
        batch_paths = all_image_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            images.append(img)

        inputs = feature_extractor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    y_true = np.array(all_true_labels)
    y_pred = np.array(all_preds)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[label2id["artificial"], label2id["human"]])

    print(f"\nAccuracy: {acc:.4f} ({int(acc * len(y_true))}/{len(y_true)})")
    print("\nConfusion Matrix (rows = true, cols = pred):")
    print("            artificial  human")
    print(f"artificial    {cm[0][0]:>6}    {cm[0][1]:>6}")
    print(f"human         {cm[1][0]:>6}    {cm[1][1]:>6}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["artificial", "human"]))


if __name__ == "__main__":
    main()