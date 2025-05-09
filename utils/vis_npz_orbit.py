import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

LABEL_COLORS = {
    0: [0.7, 0.7, 0.7],  # background
    1: [1.0, 0.2, 0.2],  # garment
}

def visualize_orbit(npz_path, save_path='orbit.gif', n_frames=60):
    data = np.load(npz_path, allow_pickle=True)
    points = data['points']              # (N, 3)
    pred = data['pred'].astype(int)     # (N,)

    colors = np.array([LABEL_COLORS[p] for p in pred])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.1)
    ax.set_axis_off()

    def update_view(i):
        ax.view_init(elev=90, azim=270 + (i * 360 / n_frames))
        return scat,

    ani = animation.FuncAnimation(fig, update_view, frames=n_frames, interval=100, blit=True)

    if save_path.endswith('.mp4'):
        ani.save(save_path, fps=12, bitrate=1800)
    else:
        ani.save(save_path, writer='pillow', fps=12)

    print(f"[✓] Saved orbit animation to {save_path}")
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='npz file to visualize')
    parser.add_argument('--output', type=str, default='orbit.gif')
    args = parser.parse_args()

    visualize_orbit(args.input, save_path=args.output)
