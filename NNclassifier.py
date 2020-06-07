import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn as sk
from Code2 import FeatureEngineering as FE
import torchvision
import tqdm
import collections

def to_cuda(elements):
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements


class FullyConnectedModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Number of input nodes (size of images)
        num_input_nodes = 20*20

        hidden_input = 32
        # Number of classes in the chars74k-lite dataset
        num_classes = 26

        # Define our model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_input_nodes, hidden_input),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_input, num_classes),
            torch.nn.ReLU()
        )

    def forward(self, x):
        # Runs a forward pass on the images
        x = x.view(-1, 20*20)
        out = self.classifier(x.float())
        return out

class Trainer:

    def __init__(self,
                 model,
                 dataloader_train,
                 dataloader_val,
                 batch_size,
                 loss_function,
                 optimizer):
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.batch_size = batch_size

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train(self, num_epochs):
        tracked_train_loss = collections.OrderedDict()
        tracked_test_loss = collections.OrderedDict()
        global_step = 0

        for epoch in range(num_epochs):
            avg_loss = 0

            for batch_it, (images, target) in enumerate(
                    tqdm.tqdm(self.dataloader_train,
                              desc=f"Training epoch {epoch}")):
                # images has shape: [batch_size, 1, 20, 20]
                # target has shape: [batch_size]

                batch_it, (images, target) = to_cuda([batch_it, (images, target)])

                # Perform forward pass
                logits = self.model(images)

                # Compute loss
                loss = self.loss_function(logits, target)

                avg_loss += loss.detach().item()
                # Perform backpropagation
                loss.backward()

                # Update our parameters with gradient descent
                self.optimizer.step()

                # Reset our model parameter gradients to 0
                self.optimizer.zero_grad()

                # Track the average loss for every 500th image
                if batch_it % (500//self.batch_size) == 0 and batch_it != 0:
                    avg_loss /= (500//self.batch_size)
                    tracked_train_loss[global_step] = avg_loss
                    avg_loss = 0
                global_step += self.batch_size
            # Compute loss and accuracy on the test set
            test_loss, test_acc = compute_loss_and_accuracy(
                self.dataloader_val, self.model, self.loss_function
            )
            tracked_test_loss[global_step] = test_loss
        return tracked_train_loss, tracked_test_loss



def plot_loss(loss_dict, label):
    global_steps = list(loss_dict.keys())
    loss = list(loss_dict.values())
    plt.plot(global_steps, loss, label=label)

def compute_loss_and_accuracy(dataloader, model, loss_function):
    """
    Computes the total loss and accuracy over the whole dataloader
    Args:
        dataloder: Test dataloader
        model: torch.nn.Module
        loss_function: The loss criterion, e.g: nn.CrossEntropyLoss()
    Returns:
        [loss_avg, accuracy]: both scalar.
    """
    model.eval()
    # Tracking variables
    loss_avg = 0
    total_correct = 0
    total_images = 0
    total_steps = 0
    with torch.no_grad():  # No need to compute gradient when testing
        for (X_batch, Y_batch) in dataloader:
            # Forward pass the images through our model
            output_probs = model(X_batch)
            # Compute loss
            loss = loss_function(output_probs, Y_batch)

            # Predicted class is the max index over the column dimension
            predictions = output_probs.argmax(dim=1).squeeze()
            Y_batch = Y_batch.squeeze()

            # Update tracking variables
            loss_avg += loss.item()
            total_steps += 1
            total_correct += (predictions == Y_batch).sum().item()
            #print("Prediction:")
            #print(predictions,Y_batch)
            total_images += predictions.shape[0]
    model.train()
    loss_avg = loss_avg / total_steps
    accuracy = total_correct / total_images
    return loss_avg, accuracy

class MyData(torch.utils.data.Dataset):

    def __init__(self,set):
        self.samples = []
        for i,elements in enumerate(set):
            for element in elements:
                self.samples += [(element,i)]


    def __getitem__(self,idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

def load_dataset(batch_size, fe, image_transform):

    dataset_train = MyData(fe.training_set)

    dataset_test = MyData(fe.test_set)


    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False
    )
    return dataloader_train, dataloader_test








if __name__ =="__main__":

    # ### Hyperparameters & Loss function

    # Hyperparameters
    batch_size = 32
    learning_rate = .11
    num_epochs = 5

    fe = FE()
    fe.HOG()
    fe.scaling()
    fe.splitDataset()

    # Use CrossEntropyLoss for multi-class classification
    loss_function = torch.nn.CrossEntropyLoss()

    # Model definition
    model = to_cuda(FullyConnectedModel())

    # Define optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    image_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor()
      
    ])
    dataloader_train, dataloader_val = load_dataset(batch_size, fe, image_transform=image_transform)
    

    trainer = Trainer(
      model=model,
      dataloader_train=dataloader_train,
      dataloader_val=dataloader_val,
      batch_size=batch_size,
      loss_function=loss_function,
      optimizer=optimizer
    )
    train_loss_dict, val_loss_dict = trainer.train(num_epochs)

    

    
    final_loss, final_acc = compute_loss_and_accuracy(dataloader_val, model, loss_function)
    print(f"Final Test Cross Entropy Loss: {final_loss}. Final Test accuracy: {final_acc}")
