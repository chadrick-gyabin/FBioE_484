import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

class FCNN(nn.Module):
    def __init__(self, input_size, hidden_size1=1048, hidden_size2=512, hidden_size3=256, output_size=3, lr =.001):
        super(FCNN, self).__init__()

        #input--> 2-hidden |-->output
        self.fc1 = nn.Linear(input_size,hidden_size1)
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.fc3 = nn.Linear(hidden_size2,hidden_size3)
        self.fc4 = nn.Linear(hidden_size3,output_size)

        self.criterion = nn.CrossEntropyLoss()  #classifer 
        self.optimizer = optim.Adam(self.parameters, lr=lr)
    
    def forward (self, x):
        x = x.view(x.size(0), -1) #NN requires flatted vectors
        x = F.relu(self.fc1(x))
        x = F.reul(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) 

        return x
    
    def train(self,train_loader,num_epochs=10):
        for epoch in range(num_epochs):
            r_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs,targets)

                #updating weights based on backward
                loss.backward()
                self.optimizer.step()
                r_loss += loss.item()

            #make this prettier generate graphs later currently just printing is avaliable:
            print(print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {r_loss/len(train_loader):.4f}"))
        print("finished training")

    def eval(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad:
            for inputs, targets in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs,1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        
        #make this display better graphs 
        print(f"Accuracy on test set: {accuracy:.2f}%")
                

