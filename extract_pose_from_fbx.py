import bpy
import numpy as np
import os

# === SETTINGS ===
FBX_PATH = "/data2/jaesung/CloSe-Reim/mixamo/Ch07_nonPBR.fbx"  # update this
OUTPUT_PATH = "/data2/jaesung/CloSe-Reim/mixamo/output_pose.npz"
# Optional: limit to SMPL-like joints (map as needed)
JOINT_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"
]  # ← up to 24

# === FBX LOAD ===
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=FBX_PATH)

# Find armature (skeleton)
armature = None
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        armature = obj
        break

if armature is None:
    raise RuntimeError("No armature found in FBX.")

bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode='POSE')

pose_vector = []
for joint in JOINT_NAMES:
    bone = armature.pose.bones.get(joint)
    if bone is None:
        pose_vector.extend([0.0, 0.0, 0.0])
        continue

    # Local rotation in axis-angle
    quat = bone.rotation_quaternion.copy()
    axis, angle = quat.to_axis_angle()
    pose_vector.extend([axis[0] * angle, axis[1] * angle, axis[2] * angle])

pose_vector = np.array(pose_vector, dtype=np.float32)
np.savez(OUTPUT_PATH, pose=pose_vector)
print(f"[✓] Saved pose vector ({len(pose_vector)}D) to {OUTPUT_PATH}")
