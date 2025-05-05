import torch
from models.FCN import FCNN
from models.Amber import ConvAutoencoder, KMeansClusterer, SVMclassifier
from utils import get_loaders,get_true_labels
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

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
    autoencoder = ConvAutoencoder()  # Instantiate the autoencoder
    train_accuracy = autoencoder.train_model(train_loader, epochs=epochs, lr=lr)  # Train the autoencoder and get training accuracy

    print("\nClustering with KMeans…")
    clusterer = KMeansClusterer(autoencoder, device=device)
    clusterer.fit(train_loader)  # Fit the KMeans clusterer on the training data

    print("\nTraining SVM…")
    svm = SVMclassifier(clusterer)
    svm.train(train_loader)  # Train the SVM classifier

    print("\nEvaluating with SVM…")
    predictions = svm.predict(test_loader)  # Predict on the test data
    
    # Convert the predictions to numpy for easy evaluation
    predictions = predictions.astype(int)  # Ensure predictions are integers (as labels)
    
    # Compute accuracy and other evaluation metrics
    y_true = get_true_labels(test_loader)  # Get the true labels for the test set
    
    test_accuracy = accuracy_score(y_true, predictions)  # Testing accuracy
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(classification_report(y_true, predictions))

    # Confusion matrix
    cm = confusion_matrix(y_true, predictions)
    print("Confusion Matrix:")
    print(cm)

    return train_accuracy, test_accuracy, predictions



def main():
    
    # Specify the directories for train and test data directly here 
    train_dir = r"C:/Users/Chaddy/OneDrive/Desktop/BioE '26/BioE 484/Final_Project/FBioE_484/images/train"
    test_dir = r"C:/Users/Chaddy/OneDrive/Desktop/BioE '26/BioE 484/Final_Project/FBioE_484/images/test"
    
    epochs = 5  # Set the number of epochs
    lr = 1e-3    # Set the learning rate
    batch_size = 32  # Set the batch size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = get_loaders(train_dir, test_dir, batch_size=batch_size)
    
    # Choose the model to evaluate (either 'fcn' or 'amber')
    model_choice = 'amber'  # Change this to 'fcn' to evaluate FCN
    
    if model_choice == 'fcn':
        evaluate_fcn(train_loader, test_loader, device, epochs=epochs, lr=lr)
    elif model_choice == 'amber':
        evaluate_amber(train_loader, test_loader, device, epochs=epochs, lr=lr)
    else:
        print("Invalid model choice. Please select 'fcn' or 'amber'.")

if __name__ == "__main__":
    main()
