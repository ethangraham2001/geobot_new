from data.image_loader import DataLoader
from data.image_display import ImageDisplay

img_loader = DataLoader()
img_loader.summarize()
train_loader, val_loader, test_loader = img_loader.get_loaders()

data_iter = iter(train_loader)
images, labels = next(data_iter)

img = images[0]

image_display = ImageDisplay()
image_display.display_image(img)