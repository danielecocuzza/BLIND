import os, glob, csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from train_all import SiameseNet


def test_stain_dataset3(
    model_path="models/siamese_all_acquisitions_contrastive_v1.pth",
    root_gallery="dataset/FIRST_ACQUISITION",
    root_noise="dataset/NOISE/STAIN",
    device=None,
    csv_path="evaluations/retrieval_ranks_stain_all.csv",
    topk_values=(1, 3, 5, 10)
):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    net = SiameseNet(embed_dim=512).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    print("Loaded model from", model_path)

    tf = transforms.ToTensor()

    # =========================
    # GALLERY
    # =========================
    def compute_gallery(folder):
        paths = sorted(
            glob.glob(os.path.join(folder, "*.png")),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )

        E, Y = [], []

        with torch.no_grad():
            for p in paths:
                img = tf(Image.open(p).convert("L")).unsqueeze(0).to(device)
                e = net.embedding(img).cpu().numpy()
                E.append(e)
                Y.append(int(os.path.basename(p).split('.')[0]))

        return np.vstack(E), np.array(Y), paths

    # =========================
    # QUERY
    # =========================
    def compute_query(folder):
        paths = sorted(
            glob.glob(os.path.join(folder, "*.png")),
            key=lambda x: (
                int(os.path.basename(x).split('_')[0]),
                int(os.path.basename(x).split('_')[1].split('.')[0])
            )
        )

        E, Y = [], []

        with torch.no_grad():
            for p in paths:
                img = tf(Image.open(p).convert("L")).unsqueeze(0).to(device)
                e = net.embedding(img).cpu().numpy()
                E.append(e)

                label = int(os.path.basename(p).split('_')[0])
                Y.append(label)

        return np.vstack(E), np.array(Y), paths

    # =========================
    # GALLERY EMBEDDINGS
    # =========================
    E_g, Y_g, P_g = compute_gallery(root_gallery)

    total_all = 0
    correct_all = 0

    ranks = []
    topk_hits_global = {k: 0 for k in topk_values}

    # =========================
    # LOOP LIVELLI
    # =========================
    for level in range(1, 10):

        folder_level = os.path.join(root_noise, str(level))
        if not os.path.exists(folder_level):
            continue

        E_q, Y_q, P_q = compute_query(folder_level)
        sims = E_q @ E_g.T

        total = 0
        correct = 0
        topk_hits_level = {k: 0 for k in topk_values}

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

            ranks.append((level, os.path.basename(P_q[i]), y, rank))

            for k in topk_values:
                if rank is not None and rank <= k:
                    topk_hits_level[k] += 1
                    topk_hits_global[k] += 1

        acc_level = correct / total if total > 0 else 0

        print(f"\n==== Livello {level} ====")
        print(f"Accuracy: {acc_level*100:.2f}% (hit={correct}, miss={total-correct})")

        for k in topk_values:
            acc_k = topk_hits_level[k] / total if total > 0 else 0
            print(f"Top-{k}: {acc_k*100:.2f}%")

        total_all += total
        correct_all += correct

    # =========================
    # GLOBAL
    # =========================
    acc_global = correct_all / total_all if total_all > 0 else 0

    print("\n===== RISULTATI GLOBALI =====")
    print(f"Top-1 Accuracy: {acc_global*100:.2f}% (hit={correct_all}, miss={total_all-correct_all})")

    for k in topk_values:
        acc_k = topk_hits_global[k] / total_all if total_all > 0 else 0
        print(f"Top-{k}: {acc_k*100:.2f}%")

    # =========================
    # SAVE CSV
    # =========================
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stain_level", "query_image", "label", "rank_position"])
        writer.writerows(ranks)

    print(f"\nCSV salvato in: {csv_path}")


if __name__ == "__main__":
    test_stain_dataset3()