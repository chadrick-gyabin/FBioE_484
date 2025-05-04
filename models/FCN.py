import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

class FCNN(nn.Module):
    def __init__(self, input_size, hidden_size1=1048, hidden_size2=512, hidden_size3=256, output_size=3, lr=.001):
        super(FCNN, self).__init__()

        # Input--> 2-hidden layers --> Output
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

        self.criterion = nn.CrossEntropyLoss()  # Cross-Entropy loss for classification
        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # Adam optimizer

    def forward(self, x):
        # Ensure input is on the same device as the model's parameters
        x = x.to(self.fc1.weight.device)  # Move input to the same device as the model's parameters
        x = x.view(x.size(0), -1)  # Flatten the input to be a 2D tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Apply ReLU activation function
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer

        return x

    def train_model(self, train_loader, device, num_epochs=10):
        self.to(device)  # Move model to the specified device (GPU or CPU)
        self.train()  # Set the model to training mode
        
        for epoch in range(num_epochs):
            r_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move data to the same device as the model
                
                self.optimizer.zero_grad()  # Zero the gradients
                outputs = self(inputs)  # Forward pass
                loss = self.criterion(outputs, targets)  # Compute loss

                # Backpropagation and optimization
                loss.backward()
                self.optimizer.step()
                r_loss += loss.item()

            # Print the average loss for this epoch
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {r_loss / len(train_loader):.4f}")
        print("Finished training")

    def evaluate(self, test_loader, device):
        self.to(device)  # Ensure the model is on the same device as the data
        self.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient computation during evaluation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move data to the same device as the model
                outputs = self(inputs)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy on test set: {accuracy:.2f}%")
