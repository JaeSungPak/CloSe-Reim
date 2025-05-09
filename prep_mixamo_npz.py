import os
import numpy as np
import trimesh
import argparse
from pytorch3d.io import load_obj
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import estimate_pointcloud_normals
import torch

LABEL_CATEGORIES = [
    'Hat', 'Body', 'Shirt', 'TShirt', 'Vest', 'Coat', 'Dress', 'Skirt',
    'Pants', 'ShortPants', 'Shoes', 'Hoodies', 'Hair', 'Swimwear',
    'Underwear', 'Scarf', 'Jumpsuits', 'Jacket'
]

label2index = {name: idx for idx, name in enumerate(LABEL_CATEGORIES)}

def compute_normals(vertices):
    pc = Pointclouds(points=[torch.tensor(vertices).float()])
    normals = estimate_pointcloud_normals(pc, neighborhood_size=50)[0]
    return normals.numpy()

def main(obj_path, save_path, garment_class):
    print(f"[INFO] Loading mesh: {obj_path}")
    mesh = trimesh.load(obj_path, process=False)
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)
    colors = np.ones_like(vertices, dtype=np.float32)  # white RGB (N, 3)
    normals = mesh.vertex_normals.astype(np.float32)

    if normals.shape[0] != vertices.shape[0] or normals.ndim != 2:
        normals = compute_normals(vertices)

    # Normalize mesh
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    center = (mesh.bounds[1] + mesh.bounds[0]) / 2
    vertices = (vertices - center) / total_size

    # Pose: dummy T-pose (72 zeros), betas and trans are also zero
    pose = np.zeros((72,), dtype=np.float32)
    betas = np.zeros((10,), dtype=np.float32)
    trans = np.zeros((3,), dtype=np.float32)

    garments = np.zeros((18,), dtype=np.float32)
    for name in garment_class:
        if name in label2index:
            garments[label2index[name]] = 1

    print("[✓] Saving to:", save_path)
    np.savez(
        save_path,
        points=vertices,
        normals=normals,
        colors=colors,
        faces=faces,
        pose=pose,
        betas=betas,
        trans=trans,
        garments=garments,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_obj', type=str, required=True, help='Path to .obj file')
    parser.add_argument('--output_npz', type=str, required=True, help='Output npz path')
    parser.add_argument('--garment_class', nargs='+', default=['TShirt', 'Pants'])
    args = parser.parse_args()

    main(args.input_obj, args.output_npz, args.garment_class)
