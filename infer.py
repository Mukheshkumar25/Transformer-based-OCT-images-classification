import torch
from torchvision import transforms
from PIL import Image
from mobilevit_model import MobileViT_XS

# class index to label mapping
idx_to_class = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 1. Load model
model = MobileViT_XS(num_classes=4).to(device)

checkpoint = torch.load("mobilevit_xs_best.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Loaded model with Val Acc:", checkpoint.get("val_acc", "N/A"))

# 2. Define transform (same as training/test)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict_image(img_path: str):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_class = idx_to_class[pred_idx]
        confidence = probs[0, pred_idx].item()

    return pred_class, confidence

if __name__ == "__main__":
    # TODO: put a real path from your local OCT images
    test_image_path = "image.png"
    pred, conf = predict_image(test_image_path)
    print(f"Prediction: {pred} ({conf*100:.2f}% confidence)")
