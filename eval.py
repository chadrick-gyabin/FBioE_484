import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import Amber, FCN

def get_loaders(train_dir, test_dir, batch_size=32, image_size=(512, 512)):
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    train_ds = ImageFolder(train_dir, transform=transform)
    test_ds  = ImageFolder(test_dir,  transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def evaluate(y_true, y_pred):
    """Return dict of weighted accuracy, precision, recall, f1."""
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1 Score":  f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

def plot_metrics(metrics_dict, save_path="metrics_comparison.png"):
    """
    metrics_dict = {
      "FCN":   {"Accuracy":…, "Precision":…, …},
      "Amber": {"Accuracy":…, "Precision":…, …},
    }
    """
    models = list(metrics_dict.keys())
    metrics = list(next(iter(metrics_dict.values())).keys())
    n = len(metrics)
    x = np.arange(n)
    width = 0.35

    plt.figure()
    for i, model in enumerate(models):
        vals = [metrics_dict[model][m] for m in metrics]
        plt.bar(x + i*width, vals, width, label=model)
    plt.xticks(x + width*(len(models)-1)/2, metrics)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

#edit as you may 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",  required=True)
    parser.add_argument("--test_dir",   required=True)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(
        args.train_dir, args.test_dir,
        batch_size=args.batch_size
    )

    # FCN 
    print("\nTraining FCN…")
    input_size = 3 * 512 * 512
    fcn = FCN(input_size=input_size, lr=args.lr).to(device)
    fcn.train(train_loader, num_epochs=args.epochs)

    print("Evaluating FCN…")
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outs = fcn(imgs)
            preds = outs.argmax(dim=1).cpu().tolist()
            y_pred += preds
            y_true += labels.tolist()
    fcn_metrics = evaluate(y_true, y_pred)
    print("FCN:", fcn_metrics)

    # Amber
    print("\nTraining Amber AE…")
    amber = Amber(img_size=512, latent_dim=16).to(device)
    amber.train_model(
        train_folder=args.train_dir,
        epochs=args.epochs,
        lr=args.lr,
        log_interval=100,
    )

    print("Clustering & SVM fitting…")
    amber.fit(dataset_dir=args.train_dir, n_clusters=len(train_loader.dataset.classes))

    print("Evaluating Amber…")
    preds = amber.predict(args.test_dir)
    
    # preds are cluster‐ids 0/1/2; map to integers 0/1/2
    y_pred_amber = [int(p) for p in preds]
    y_true_amber = [int(t) for t in test_loader.dataset.targets]
    amber_metrics = evaluate(y_true_amber, y_pred_amber)
    print("Amber:", amber_metrics)

    # --- Plot comparative bar chart ---
    plot_metrics({"FCN": fcn_metrics, "Amber": amber_metrics})

if __name__ == "__main__":
    main()
