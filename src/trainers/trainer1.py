import torch
import torch.optim as optim
from colorama import Fore, Style, Back
import time

from . import LEARNING_RATE, NUM_EPOCHS, CRITERION
DEVICE = torch.device("cuda")

class ModelTrainer1:
    """
    Trains a model
    """
    save_location = 'model.pth'

    def __init__(self, model, train_loader, lr=LEARNING_RATE, 
                 num_epochs=NUM_EPOCHS, criterion=CRITERION):
        self.model = model
        self.train_loader = train_loader
        self.lr = lr
        self.num_epochs = num_epochs
        self.criterion = criterion

    def train(self):
        """
        Trains the class' model
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        total_time = 0
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            batch_count = 0
            total_samples = 0
            correct_predictions = 0 
            start_time = time.time()

            print(Back.WHITE + Fore.BLACK + Style.BRIGHT)
            print(f"starting epoch: [{epoch}/{self.num_epochs}]" + Back.RESET + Fore.RESET + Style.RESET_ALL)

            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_count += 1

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0) 

                correct_predictions += (predicted==labels).sum().item()

                if batch_count % 10 == 0:
                    print(Style.DIM + f'   -->batch={batch_count}' + Style.RESET_ALL)

            # save model state
            torch.save(self.model.state_dict(), self.save_location)
            print(Style.BRIGHT + f'--> model saved to {self.save_location}!' + Style.RESET_ALL)

            accuracy = correct_predictions / total_samples
            print(Style.BRIGHT + f'--> accuracy = {accuracy}%' + Style.RESET_ALL)
            
            end_time = time.time()
            iteration_time = end_time - start_time
            total_time += iteration_time
            print(Fore.CYAN + Style.DIM + f"--> Epoch took {iteration_time}. loss = {loss.item()}")

