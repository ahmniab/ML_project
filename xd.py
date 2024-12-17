import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , label_binarize
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

data_root = './data/'

train_data = datasets.STL10(root=data_root, split='train')
test_data = datasets.STL10(root=data_root, split='test')


transform = transforms.ToTensor()
num_of_rows = 3
classes_count = len(train_data.classes)

fig, axes = plt.subplots(num_of_rows, classes_count, figsize=(15, 5))

for row in range(3):
    for i in range(classes_count):
        img, label = train_data[i + row*classes_count]
        tensor_img = transform(img)

        tensor_img = torch.clamp(tensor_img, 0, 1)

        # Convert to numpy for display
        img_to_show = tensor_img.permute(1, 2, 0).numpy()

        # Display image
        axes[row, i].imshow(img_to_show)
        axes[row, i].axis('off')  # Hide axes
        axes[row, i].set_title(f"{train_data.classes[label]}")

plt.tight_layout()
plt.show()
