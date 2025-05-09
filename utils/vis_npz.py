import numpy as np
import matplotlib.pyplot as plt
import os

LABEL_COLORS = {
    0: [0.7, 0.7, 0.7],  # background
    1: [1.0, 0.2, 0.2],  # garment
}

def visualize_image(npz_path, save_path='vis.png'):
    data = np.load(npz_path, allow_pickle=True)
    points = data['points']              # (N, 3)
    pred = data['pred'].astype(int)     # (N,)

    colors = np.array([LABEL_COLORS[p] for p in pred])
    num_foreground = int((pred == 1).sum())
    print(f"[✓] Foreground points (label=1): {num_foreground} / {len(pred)}")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.1)
    ax.view_init(elev=90, azim=270)  # front view
    ax.set_axis_off()

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"[✓] Saved static view to {save_path}")
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='npz file to visualize')
    parser.add_argument('--output', type=str, default='vis.png')
    args = parser.parse_args()

    visualize_image(args.input, save_path=args.output)
