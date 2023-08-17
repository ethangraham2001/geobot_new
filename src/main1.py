import torch
from trainers.trainer1 import ModelTrainer1
from models.cnn_model1 import CNN_1
from data.image_loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device is {device}")

data_loader = DataLoader()
train_loader, val_loader, test_loader = data_loader.get_loaders()

model = CNN_1()

trainer = ModelTrainer1(
    model=model,
    train_loader=train_loader
)

trainer.train()