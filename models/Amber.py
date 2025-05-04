import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import numpy as np

class ConvAutoencoder(nn.Module):
    def __init__(self, img_channels=3, img_size=512, latent_dim=16):
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

    def train_model(self, train_loader, epochs=50, lr=1e-3, log_interval=100):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for batch_idx, batch in enumerate(train_loader, 1):
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                imgs = imgs.to(self.device)

                optimizer.zero_grad()
                recon = self(imgs)
                loss = criterion(recon, imgs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch_idx % log_interval == 0:
                    avg = running_loss / log_interval
                    print(f"[Epoch {epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)}  Loss: {avg:.6f}")
                    running_loss = 0.0

        print("Training complete.")

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
                features_list.append(latent_vectors.cpu().numpy())

        return np.concatenate(features_list, axis=0)
    
    def fit(self, data_loader, n_clusters=3, **kmeans_kwargs):
        X = self.latent_extractor(data_loader)
        self.kmeans = KMeans(n_clusters=n_clusters, **kmeans_kwargs).fit(X)
        self.labels_ = self.kmeans.labels_
        self.cluster_centers_ = self.kmeans.cluster_centers_
        return self

class SVMclassifier:
    def __init__(self, clusterer, kernel='linear'):
        self.clusterer = clusterer
        self.svm = SVC(kernel=kernel, decision_function_shape='ovr')
    
    def train(self, data_loader):
        latent_features = self.clusterer.latent_extractor(data_loader)
        labels = self.clusterer.labels_
        self.svm.fit(latent_features, labels)
        print("SVM training done")

    def predict(self, data_loader):
        latent_features = self.clusterer.latent_extractor(data_loader)
        predictions = self.svm.predict(latent_features)
        return predictions
