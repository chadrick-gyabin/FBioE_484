import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from sklearn.cluster import KMeans 

class ConvAutoencoder(nn.Module):
    def __init__(self, img_channels =3,img_size = 512, latent_dim= 16, device = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim

        #Enconder 512 --> 256 --> 128 -->64 
        self.enc = nn.Sequential(
            nn.Conv2d(img_channels, 32,3, stride=2, padding=1), # [32,256,256]
            nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=2,padding=1), #[64,128,128]
            nn.ReLU(True),
            nn.Conv2d(64,127,3,stride=2,padding=1), #[128,64,64]
            nn.ReLU(True),
            nn.Flatten(), #[128*64*64,1]
            nn.Linear(128*(img_size//8)*(img_size//8),latent_dim),
        )
        
        #Decoder: 64-->128-->256-->512
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128*(img_size//8)*(img_size//8)),
            nn.ReLU(True),
            nn.Unflatten(1, (128, img_size//8, img_size//8)),       # (128,64,64)
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # (64,128,128)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # (32,256,256)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, img_channels, 3, stride=2, padding=1, output_padding=1),  # (3,512,512)
            nn.Sigmoid(),  # output in [0,1]
        )
        
        self.to(self.device) 

    def forward(self,x):
        return self.dec(self.enc(x))

    def encode(self,x):
        self.eval()
        with torch.no_grad():
            return self.enc(x.to(self.device))
    
    def decode(self, z):
        self.eval()
        with torch.no_grad():
            return self.dec(z.to(self.device))  
    
    def train_model(self, dataloader, epochs=50, lr=1e-3, val_loader=None, log_interval=100):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(1, epochs+1):
            self.train()
            running_loss = 0.0
            for batch_idx, batch in enumerate(dataloader, 1):
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
                    print(f"[Epoch {epoch}/{epochs}] Batch {batch_idx}/{len(dataloader)}  Loss: {avg:.6f}")
                    running_loss = 0.0

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                        imgs = imgs.to(self.device)
                        val_loss += criterion(self(imgs), imgs).item()
                val_loss /= len(val_loader)
                print(f"--- Epoch {epoch} Validation Loss: {val_loss:.6f}")

        print("Training complete.")

class KMeansClusterer:
    """
    Just adding a KMeansClustering on the latent space 
    """
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.kmeans = None
        self.labels_ = None
        self.cluster_centers_ = None
    
    def latent_extractor(self,data_loader) -> np.ndarray: 
        """
        all images are encoded into the latent space then turned from tensor to np appary
        """
        self.model.eval()
        features_list = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(self.device)
                latent_vectors = self.model.encode(images)   # was `codes`
                features_list.append(latent_vectors.cpu().numpy())

        return np.concatenate(features_list, axis=0)
    
    def fit(self, data_loader, n_clusters=3, **kmeans_kwargs):
        """
        Does 2 things: 
        1. Extracts latent features
        2. Fit K-Means with 3 clusters (for the number of classes)
        """
        X = self.latent_extractor(data_loader)
        self.kmeans = KMeans(n_clusters=n_clusters, **kmeans_kwargs).fit(X)
        self.labels_ = self.kmeans.labels_
        self.cluster_centers_ = self.kmeans.cluster_centers_
        return self
    
   #finish SVM and other
    