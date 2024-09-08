import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor
import hashlib
import time
import numpy as np
import os
import random
import pickle

# Assume this is the path to your SSD cache directory
CACHE_DIR = "/home/xshadow/Datasets/cache_dir"



def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)



class FeatureDistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_features, teacher_features):
        return self.mse_loss(student_features, teacher_features)

class CachedFeatureDataset(Dataset):
    def __init__(self, image_ids, images, labels, teacher_model, cache_dir):
        self.image_ids = image_ids
        self.images = images
        self.labels = labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model = teacher_model.to(self.device)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = self.images[idx]
        label = self.labels[idx]

        cache_file = os.path.join(self.cache_dir, f"{image_id}.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                teacher_features = torch.from_numpy(pickle.load(f))

        else:
            with torch.no_grad():
                image = image.to(self.device)
                teacher_output = self.teacher_model(image.unsqueeze(0), output_hidden_states=True)
                teacher_features = teacher_output.hidden_states[-1].squeeze(0)

            with open(cache_file, 'wb') as f:
                teacher_features = teacher_features.cpu().numpy()
                pickle.dump(teacher_features, f)

        return image, label, teacher_features

class DistilledViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = ViTModel.from_pretrained('google/vit-large-patch16-224')
        self.student = ViTModel(ViTConfig.from_pretrained('google/vit-base-patch16-224'))
        self.connector = torch.nn.Linear(768,1024)

        # Freeze teacher weights
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x, teacher_features):
        # Student forward pass
        student_output = self.student(x, output_hidden_states=True)
        student_features = student_output.hidden_states[-1]

        return student_features, teacher_features

def generate_fake_data(num_samples, image_size):
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    fake_images = torch.rand(num_samples, 3, image_size, image_size)
    fake_labels = torch.randint(0, 1000, (num_samples,))

    # Preprocess the images
    inputs = feature_extractor(images=fake_images, return_tensors="pt")
    pixel_values = inputs.pixel_values

    # Generate fake image IDs
    image_ids = [hashlib.md5(img.numpy().tobytes()).hexdigest() for img in pixel_values]

    return image_ids, pixel_values, fake_labels

def test_performance(model, dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = FeatureDistillationLoss()
    optimizer = torch.optim.Adam(model.student.parameters(), lr=1e-4)

    start_time = time.time()

    for epoch in range(num_epochs):
        for inputs, labels, teacher_features in dataloader:
            inputs = inputs.to(device)
            teacher_features = teacher_features.to(device)

            optimizer.zero_grad()
            student_features, _ = model(inputs, teacher_features)
            student_features = model.connector(student_features)

            loss = criterion(student_features, teacher_features)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} completed")

    end_time = time.time()
    total_time = end_time - start_time

    return total_time

# Test parameters
num_samples = 500
image_size = 224
batch_size = 16
num_epochs = 3


set_seed(42)

# Generate fake data
image_ids, pixel_values, fake_labels = generate_fake_data(num_samples, image_size)

# Create model
model = DistilledViT()

# Create dataset with SSD caching
dataset = CachedFeatureDataset(image_ids, pixel_values, fake_labels, model.teacher, CACHE_DIR)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test performance
print("Testing with SSD caching...")
time_with_ssd_cache = test_performance(model, dataloader, num_epochs)

print(f"\nTime taken with SSD cache: {time_with_ssd_cache:.2f} seconds")

# To compare with no caching, you would need to run the script twice:
# Once with an empty cache directory, and once with a populated cache directory
