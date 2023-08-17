from data.image_loader import DataLoader
from data.image_display import ImageDisplay
from data.helpers import nbr_to_country

img_loader = DataLoader()
img_loader.summarize()
train_loader, val_loader, test_loader = img_loader.get_loaders()

data_iter = iter(train_loader)
images, labels = next(data_iter)


img = images[0]

label = labels[0].item()
label_str = nbr_to_country(label)+'-'+str(label)

image_display = ImageDisplay()
image_display.display_image(img, label_str)