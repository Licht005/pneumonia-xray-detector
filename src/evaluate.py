import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import get_datasets  # Assumes dataset.py defines get_datasets()
from model import get_resnet18_model

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = get_resnet18_model()
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pneumonia_model.pth"))

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

print(f"Loading model from: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load datasets
_, _, test_dataset = get_datasets()
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluation loop
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Print classification report
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Confusion matrix plot
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


#run file from src/evaluate.py