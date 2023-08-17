from . import LEARNING_RATE, NUM_EPOCHS, CRITERION

class ModelTrainer1:
    """
    Trains a model
    """

    def __init__(self, model, train_loader, lr=LEARNING_RATE, 
                 num_epochs=NUM_EPOCHS, criterion=CRITERION):
        self.model = model
        self.train_loader = train_loader
        self.lr = lr
        self.num_epochs = num_epochs
        self.criterion = criterion

    def train(self):
        """
        Trains the model
        """
        for epoch in range(self.num_epochs):
            for images, labels in self.train_loader:
                print(labels.shape)