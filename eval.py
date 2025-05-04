import torch
from models.FCN import FCNN
from models.Amber import ConvAutoencoder, KMeansClusterer, SVMclassifier
from utils import get_loaders  # Assuming get_loaders is in utils.py

# Function to evaluate the FCN model
def evaluate_fcn(train_loader, test_loader, device, epochs=50, lr=1e-3):
    print("\nTraining FCN…")
    input_size = 3 * 512 * 512  # Assuming image size 512x512 and 3 channels
    fcn = FCNN(input_size=input_size, lr=lr).to(device)
    fcn.train_model(train_loader, device, num_epochs=epochs)
    fcn.evaluate(test_loader, device)

# Function to evaluate the Amber model (Autoencoder + KMeans + SVM)
def evaluate_amber(train_loader, test_loader, device, epochs=50, lr=1e-3):
    print("\nTraining Amber Autoencoder…")
    autoencoder = ConvAutoencoder()  # 
    autoencoder.train_model(train_loader, epochs=epochs, lr=lr)

    print("\nClustering with KMeans…")
    clusterer = KMeansClusterer(autoencoder, device=device)
    clusterer.fit(train_loader)

    print("\nTraining SVM…")
    svm = SVMclassifier(clusterer)
    svm.train(train_loader)

    print("\nEvaluating with SVM…")
    predictions = svm.predict(test_loader)
    # Here you can compute accuracy or other metrics using predictions
    return predictions

def main():
    # Specify the directories for train and test data directly here
    train_dir = r"C:/Users/Chaddy/OneDrive/Desktop/BioE '26/BioE 484/Final_Project/FBioE_484/images/train"
    test_dir = r"C:/Users/Chaddy/OneDrive/Desktop/BioE '26/BioE 484/Final_Project/FBioE_484/images/test"
    
    epochs = 5  # Set the number of epochs
    lr = 1e-3    # Set the learning rate
    batch_size = 32  # Set the batch size
    
    # Specify the device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get the data loaders for train and test sets
    train_loader, test_loader = get_loaders(train_dir, test_dir, batch_size=batch_size)
    
    # Choose the model to evaluate (either 'fcn' or 'amber')
    model_choice = 'fcn'  # Change this to 'fcn' to evaluate FCN
    
    if model_choice == 'fcn':
        evaluate_fcn(train_loader, test_loader, device, epochs=epochs, lr=lr)
    elif model_choice == 'amber':
        evaluate_amber(train_loader, test_loader, device, epochs=epochs, lr=lr)
    else:
        print("Invalid model choice. Please select 'fcn' or 'amber'.")

if __name__ == "__main__":
    main()
