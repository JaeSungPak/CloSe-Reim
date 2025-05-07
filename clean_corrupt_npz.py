import numpy as np
from pathlib import Path
import shutil

# 설정
data_dir = Path("./data/train")
backup_dir = data_dir / "corrupt"
backup_dir.mkdir(exist_ok=True)

def is_valid_npz(npz_path):
    try:
        data = np.load(npz_path, allow_pickle=True)

        def check_shape(arr, expected):
            return isinstance(arr, np.ndarray) and arr.shape == expected

        return (
            check_shape(data['points'], (data['points'].shape[0], 3)) and
            check_shape(data['colors'], (data['colors'].shape[0], 3)) and
            check_shape(data['normals'].squeeze(-1) if data['normals'].ndim == 3 else data['normals'], (data['colors'].shape[0], 3)) and
            data['pose'].shape == (72,) and
            data['garments'].shape == (18,)
        )

    except Exception as e:
        print(f"[X] Failed to read {npz_path}: {e}")
        return False

def clean_dataset():
    count = 0
    files = sorted(data_dir.glob("*.npz"))
    print(f"[INFO] Checking {len(files)} files...")

    for npz_file in files:
        if not is_valid_npz(npz_file):
            print(f"[!] Moving invalid file: {npz_file.name}")
            shutil.move(str(npz_file), str(backup_dir / npz_file.name))
            count += 1

    print(f"[DONE] Moved {count} corrupt files to {backup_dir}")

if __name__ == "__main__":
    clean_dataset()
