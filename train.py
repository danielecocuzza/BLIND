import os, glob, random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PairsDataset(Dataset):
    def __init__(self, root="dataset/FIRST_ACQUISITION", size=(128,640), n_pairs=5000, augment=False):
        self.paths = sorted(glob.glob(os.path.join(root, "*.png")),
                            key=lambda x: int(os.path.basename(x).split('.')[0]))
        self.labels = [int(os.path.basename(p).split('.')[0]) for p in self.paths]
        self.class_to_paths = {}
        for p,l in zip(self.paths,self.labels):
            self.class_to_paths.setdefault(l,[]).append(p)

        self.tf = transforms.ToTensor()

        self.n_pairs = n_pairs
        self.classes = list(set(self.labels))

    def __len__(self): return self.n_pairs

    def __getitem__(self, idx):
        # 50% coppie positive, 50% negative
        if random.random()<0.5:
            cls = random.choice(self.classes)
            if len(self.class_to_paths[cls]) >= 2:
                p1,p2 = random.sample(self.class_to_paths[cls], 2)
            else:
                p1=p2=self.class_to_paths[cls][0]
            label = 0.0
        else:
            cls1, cls2 = random.sample(self.classes,2)
            p1 = random.choice(self.class_to_paths[cls1])
            p2 = random.choice(self.class_to_paths[cls2])
            label = 1.0

        img1 = self.tf(Image.open(p1).convert("L"))
        img2 = self.tf(Image.open(p2).convert("L"))
        return img1, img2, torch.tensor(label, dtype=torch.float32)


class DeeperCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=3), nn.BatchNorm2d(32), nn.ELU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ELU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=2), nn.BatchNorm2d(64), nn.ELU(), nn.AvgPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2), nn.BatchNorm2d(128), nn.ELU(), nn.AvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(embed_dim),  
            nn.ELU()
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return F.normalize(x,p=2,dim=1)

class SiameseNet(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embedding = DeeperCNN(embed_dim)
    def forward(self,x1,x2):
        z1 = self.embedding(x1)
        z2 = self.embedding(x2)
        return z1,z2



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.8):
        super().__init__()
        self.margin=margin
    def forward(self,z1,z2,y):
        d = F.pairwise_distance(z1,z2)
        
        loss = (1-y)*d.pow(2) + y*F.relu(self.margin-d).pow(2)
        return loss.mean()


def train_model(root_train="dataset/FIRST_ACQUISITION",
                epochs=100, patience=5, device=None, save_path="DATASET/DATASET3/models/best_model.pth"):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Dataset e dataloader (solo training)
    ds = PairsDataset(root_train, n_pairs=7000)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    # Modello e ottimizzatore
    net = SiameseNet(embed_dim=256).to(device)
    loss_fn = ContrastiveLoss(margin=1.3)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    best_loss = float("inf")
    patience_counter = 0

    for ep in range(1, epochs + 1):
        net.train()
        running = 0.0

        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            z1, z2 = net(x1, x2)
            loss = loss_fn(z1, z2, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        avg_loss = running / len(loader)
        print(f"[Ep {ep}] train_loss={avg_loss:.5f}")

        # Early stopping basato su train loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(net.state_dict(), save_path)
            print("  ✅ New best model saved!")
        else:
            patience_counter += 1
            print(f"  ⚠️ No improvement (patience {patience_counter}/{patience})")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered!")
                break

    print(f"Training finished. Best train_loss={best_loss:.4f}")






if __name__=="__main__":
    train_model()