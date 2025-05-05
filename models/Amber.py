import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_true_labels

# Define the ConvAutoencoder class
class ConvAutoencoder(nn.Module):
    def __init__(self, img_channels=3, img_size=512, latent_dim=256):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim

        # Encoder: 512 --> 256 --> 128 --> 64 
        self.enc = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, stride=2, padding=1),  # [32,256,256]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [64,128,128]
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [128,64,64]
            nn.ReLU(True),
            nn.Flatten(),  # [128*64*64,1]
            nn.Linear(128 * (img_size // 8) * (img_size // 8), latent_dim),
        )
        
        # Decoder: 64 --> 128 --> 256 --> 512
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128 * (img_size // 8) * (img_size // 8)),
            nn.ReLU(True),
            nn.Unflatten(1, (128, img_size // 8, img_size // 8)),  # (128,64,64)
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # (64,128,128)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # (32,256,256)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, img_channels, 3, stride=2, padding=1, output_padding=1),  # (3,512,512)
            nn.Sigmoid(),  # output in [0,1]
        )
        
        self.to(self.device)

    def forward(self, x):
        return self.dec(self.enc(x))

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            return self.enc(x.to(self.device))
    
    def decode(self, z):
        self.eval()
        with torch.no_grad():
            return self.dec(z.to(self.device)) 
        
    # Methods to calculate reconstruction error, MAE
    def calculate_reconstruction_error(self, inputs, reconstructions):
        return nn.MSELoss()(reconstructions, inputs).item()

    def calculate_mae(self, inputs, reconstructions):
        return nn.L1Loss()(reconstructions, inputs).item()

    def train_model(self, train_loader, epochs=50, lr=1e-3, log_interval=100):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            running_reconstruction_error = 0.0
            running_mae = 0.0
            total_batches = 0
            
            for batch_idx, batch in enumerate(train_loader, 1):
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                imgs = imgs.to(self.device)

                optimizer.zero_grad()
                recon = self(imgs)  # Reconstructed images
                loss = criterion(recon, imgs)  # Calculate loss based on reconstruction error
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_reconstruction_error += self.calculate_reconstruction_error(imgs, recon)
                running_mae += self.calculate_mae(imgs, recon)
                total_batches += 1

                if batch_idx % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    avg_reconstruction_error = running_reconstruction_error / log_interval
                    avg_mae = running_mae / log_interval

                    print(f"[Epoch {epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)}  "
                          f"Loss: {avg_loss:.6f}, Reconstruction Error: {avg_reconstruction_error:.6f}, "
                          f"MAE: {avg_mae:.6f}")

                    # Reset metrics for the next interval
                    running_loss = 0.0
                    running_reconstruction_error = 0.0
                    running_mae = 0.0

            # Average metrics for the whole epoch
            avg_loss = running_loss / total_batches
            avg_reconstruction_error = running_reconstruction_error / total_batches
            avg_mae = running_mae / total_batches

            print(f"Epoch {epoch}/{epochs} completed. "
                  f"Avg Loss: {avg_loss:.6f}, Avg Reconstruction Error: {avg_reconstruction_error:.6f}, "
                  f"Avg MAE: {avg_mae:.6f}")

        print("Training complete.")

# Define the KMeansClusterer class
class KMeansClusterer:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.kmeans = None
        self.labels_ = None
        self.cluster_centers_ = None
    
    def latent_extractor(self, data_loader) -> np.ndarray:
        """
        Extracts all images into the latent space and returns them as a numpy array.
        """
        self.model.eval()
        features_list = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(self.device)
                latent_vectors = self.model.encode(images)
                features_list.append(tensor_to_numpy(latent_vectors))  # Convert tensor to numpy here

        return np.concatenate(features_list, axis=0)
    
    def fit(self, data_loader, n_clusters=3, **kmeans_kwargs):
        X = self.latent_extractor(data_loader)
        self.kmeans = KMeans(n_clusters=n_clusters, **kmeans_kwargs).fit(X)
        kmeans_labels = self.kmeans.labels_
    
        true_labels = get_true_labels(data_loader)  # Get true labels from data loader
        cluster_to_true_label = {}
    
        for cluster in np.unique(kmeans_labels):
            cluster_true_labels = [true_labels[i] for i in range(len(true_labels)) if kmeans_labels[i] == cluster]
            most_common_label = Counter(cluster_true_labels).most_common(1)[0][0]
            cluster_to_true_label[cluster] = most_common_label
        
        # Map KMeans labels to true labels
        self.labels_ = np.array([cluster_to_true_label[label] for label in kmeans_labels])
        self.cluster_centers_ = self.kmeans.cluster_centers_
        return self
    
class SVMclassifier:
    def __init__(self, clusterer, kernel='linear'):
        self.clusterer = clusterer
        self.svm = SVC(kernel=kernel, decision_function_shape='ovr')
        self.train = self._train_svm

    def _train_svm(self, data_loader, test_data_loader=None):
        """
        Function to train the SVM using latent features directly (without PCA).
        """
        latent_features = self.clusterer.latent_extractor(data_loader) #extract latent from autoencoder
        new_latent_features = preprocessing(latent_features) #standard latent space features 
        labels = self.clusterer.labels_ #pseudo-labels from KMeans Clustering

        #Using GridSearchCV for hyperparmater tuning
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
        # Initialize the SVM model
        svm_model = SVC()

        #GridSearchCv for hyperparameter optimization 
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1)  # Use all available CPU cores
        grid_search.fit(new_latent_features, labels)
        best_model = grid_search.best_estimator_
        self.svm = best_model

        #debugging
        print("SVM training with hyperparamter tuning done")

        if test_data_loader is not None:
            # Extract test latent features
            test_latent_features = self.clusterer.latent_extractor(test_data_loader)
            # Preprocess the test latent features
            new_test_latent_features = preprocessing(test_latent_features)
            
            # Get the true labels from the test data (real ground truth labels)
            test_labels = get_true_labels(test_data_loader)  # Get true labels from test data
            
            #Make predictions on the test set using the best model
            y_pred = self.svm.predict(new_test_latent_features)
            
            #accuracy and other evaluation metrics
            accuracy = accuracy_score(test_labels, y_pred)
            print(f"Accuracy of sklearn SVM: {accuracy:.2f}")
            print(classification_report(test_labels, y_pred))

            # Confusion matrix
            cm = confusion_matrix(test_labels, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()

            # Calculate Cohen's Kappa and Cohen's d
            kappa = cohen_kappa(test_labels, y_pred)
            print(f"Cohen's Kappa: {kappa:.2f}")
            
            cohen_d_value = cohen_d(test_labels, y_pred)
            print(f"Cohen's d: {cohen_d_value:.2f}")

    def predict(self, data_loader):
        latent_features = self.clusterer.latent_extractor(data_loader)
        predictions = self.svm.predict(latent_features)
        return predictions
        

# Preprocessing function to standardize the data
def preprocessing(latent_features):
    scaler = StandardScaler()
    standard_latent_features = scaler.fit_transform(latent_features)
    return standard_latent_features

# Cohen's d calculation
def cohen_d(y_true, y_pred):
    mean_pos = np.mean(y_true == 1)
    mean_neg = np.mean(y_true == 0)
    pooled_std = np.sqrt((np.std(y_true == 1) ** 2 + np.std(y_true == 0) ** 2) / 2)
    return (mean_pos - mean_neg) / pooled_std

# Cohen's Kappa calculation
def cohen_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()