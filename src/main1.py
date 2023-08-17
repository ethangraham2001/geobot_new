import torch
from trainers.trainer1 import ModelTrainer1
from models.cnn_model1 import CNN_1
from data.image_loader import DataLoader
from data.image_display import ImageDisplay

DEVICE = torch.device("cuda")

device = DEVICE
print(f"device is {device}")

data_loader = DataLoader()
train_loader, val_loader, test_loader = data_loader.get_loaders()

model = CNN_1().to(device=device)

trainer = ModelTrainer1(
    model=model,
    train_loader=train_loader
)

trainer.train()


# model_path = 'model.pth'
# model.load_state_dict(torch.load(model_path))
# model.summary()

# model.eval()

# val_iter = iter(val_loader)
# images, labels = next(val_iter)

# img = images[0]
# true_label = labels[0]
# display = ImageDisplay()

# print(img.shape)

# with torch.no_grad():
#     output = model.forward(img)