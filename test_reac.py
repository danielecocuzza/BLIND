import os, glob, csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from train import SiameseNet

def test_model(model_path="models/best_model.pth",
               root_gallery="dataset/FIRST_ACQUISITION",
               root_query="dataset/BRIGHTNESS/1.0/REACQUISITIONS",
               device=None,
               csv_path="dataset/evaluations/retrieval_ranks_reac.csv",
               topk_values=(1, 3, 5, 10)):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    net = SiameseNet(embed_dim=512).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    print("Loaded model from", model_path)

    tf = transforms.ToTensor()

    def compute_gallery(folder):
        paths = sorted(glob.glob(os.path.join(folder, "*.png")),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
        E, Y = [], []
        with torch.no_grad():
            for p in paths:
                img = tf(Image.open(p).convert("L")).unsqueeze(0).to(device)
                e = net.embedding(img).cpu().numpy()
                E.append(e)
                Y.append(int(os.path.basename(p).split('.')[0]))
        return np.vstack(E), np.array(Y), paths

    def compute_query(folder):
        paths = sorted(glob.glob(os.path.join(folder, "*.png")),
                       key=lambda x: (int(os.path.basename(x).split('_')[0]),
                                      int(os.path.basename(x).split('_')[1].split('.')[0])))
        E, Y = [], []
        with torch.no_grad():
            for p in paths:
                img = tf(Image.open(p).convert("L")).unsqueeze(0).to(device)
                e = net.embedding(img).cpu().numpy()
                E.append(e)
                label = int(os.path.basename(p).split('_')[0])
                Y.append(label)
        return np.vstack(E), np.array(Y), paths

    E_g, Y_g, P_g = compute_gallery(root_gallery)
    E_q, Y_q, P_q = compute_query(root_query)

    sims = E_q @ E_g.T

    total, correct = 0, 0
    ranks = []
    topk_hits = {k: 0 for k in topk_values}

    for i, y in enumerate(Y_q):
        sim_row = sims[i]
        sorted_idx = np.argsort(-sim_row)
        sorted_labels = Y_g[sorted_idx]

        pred = sorted_labels[0]
        if pred == y:
            correct += 1
        total += 1

        correct_positions = np.where(sorted_labels == y)[0]
        rank = int(correct_positions[0]) + 1 if len(correct_positions) > 0 else None
        ranks.append((os.path.basename(P_q[i]), y, rank))

        for k in topk_values:
            if rank is not None and rank <= k:
                topk_hits[k] += 1

    print(f"\nAccuracy globale = {correct/total*100:.2f}%  (hit={correct}, miss={total-correct})")

    for k in topk_values:
        acc = topk_hits[k] / total
        print(f"Top-{k} Accuracy: {acc*100:.2f}%")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_image", "label", "rank_position"])
        writer.writerows(ranks)

    print(f"\nCSV salvato in: {csv_path}")


if __name__ == "__main__":
    test_model()