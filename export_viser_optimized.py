#!/usr/bin/env python3
"""
Export viser_cache.pkl to .viser format for static web playback.
OPTIMIZED VERSION: Uses float32 and gzip compression for smaller files.

Usage:
    python export_viser_optimized.py --input /path/to/decay_0.0/viser_cache.pkl --output assets/viser/scene.viser
    python export_viser_optimized.py --input-dir /path/to/decay_outputs_streaming/uuid --output-dir assets/viser/
"""

import argparse
import pickle
import os
import glob
import time
import gzip
from pathlib import Path

import numpy as np



def export_single_pkl(
    pkl_path: str,
    output_path: str,
    fps: float = 30.0,
    point_size: float = 0.007,
    camera_convention: str = "c2w",
    camera_extra_rot: str = "none",
    center_mode: str = "none",
    world_rot: str = "none",
    coord_mode: str = "da3",
    world_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    use_gzip: bool = False,
    downsample: float = 1.0,
    use_fp16: bool = False,
    frame_step: int = 1,
    scale: float = 1.0,
    center_override: "np.ndarray | None" = None,
):
    """
    Export a single viser_cache.pkl to .viser format.

    Args:
        pkl_path: Path to viser_cache.pkl
        output_path: Output .viser file path
        fps: Playback frame rate
        point_size: Point cloud point size
        use_gzip: Whether to gzip compress the output
        downsample: Fraction of points to keep (0.0-1.0), e.g. 0.5 = keep 50%
        use_fp16: Whether to use float16 for point cloud coordinates (smaller files)
        frame_step: Take every Nth frame (e.g. 2 = keep half the frames)
        scale: Scale factor for the scene (e.g. 0.125 to shrink 8x)
    """
    import viser

    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        frames = pickle.load(f)

    if not frames:
        print(f"  Warning: No frames in {pkl_path}, skipping.")
        return False

    print(f"  Found {len(frames)} frames")

    # Frame step
    if frame_step > 1:
        frames = frames[::frame_step]
        print(f"  Frame step {frame_step}: {len(frames)} frames kept")

    # Downsampling setup
    if downsample < 1.0:
        n_points_original = frames[0]["points"].shape[0]
        n_points_keep = int(n_points_original * downsample)
        # Use consistent random indices across all frames for temporal consistency
        np.random.seed(42)
        downsample_indices = np.random.choice(n_points_original, n_points_keep, replace=False)
        downsample_indices = np.sort(downsample_indices)  # Sort for cache-friendly access
        print(f"  Downsampling: {n_points_original:,} -> {n_points_keep:,} points ({downsample*100:.0f}%)")
    else:
        downsample_indices = None

    # Create viser server on a random high port
    port = 18000 + os.getpid() % 1000
    server = viser.ViserServer(port=port, verbose=False)

    # DA3 uses -Y up; WebGL/glTF uses +Y up
    server.scene.set_up_direction("-y" if coord_mode == "da3" else "+y")

    # Give server time to initialize
    time.sleep(0.5)

    # Get serializer AFTER setting up the scene basics
    serializer = server.get_scene_serializer()

    # Frame duration
    frame_duration = 1.0 / fps

    print(f"  Processing {len(frames)} frames...")

    # Optional recentering to put the scene near origin for better default view
    # Use float32 for all calculations to save memory
    if center_override is not None:
        center_offset = center_override.astype(np.float32)
    else:
        center_offset = np.zeros(3, dtype=np.float32)
        if center_mode != "none":
            ref_pts = frames[0]["points"].astype(np.float32)
            if center_mode == "first_bbox":
                center_offset = (ref_pts.min(axis=0) + ref_pts.max(axis=0)) * 0.5
            elif center_mode == "first_mean":
                center_offset = ref_pts.mean(axis=0)

    # DA3 writes aligned world coords and c2w quats. Optional OpenCV->WebGL flip.
    def transform_opencv_to_gltf(points):
        if coord_mode == "da3":
            return points
        transformed = points.copy()
        transformed[:, 1] = -transformed[:, 1]
        transformed[:, 2] = -transformed[:, 2]
        return transformed

    def transform_camera_pos(pos):
        if coord_mode == "da3":
            return np.array(pos, dtype=np.float32)
        return np.array([pos[0], -pos[1], -pos[2]], dtype=np.float32)

    def _quat_mul(q1, q2):
        """Quaternion multiply (wxyz)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float32)

    def _quat_conj(q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z], dtype=np.float32)

    def _rotate_vec(q_wxyz, v):
        """Rotate vector v by quaternion q (wxyz)."""
        qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
        return _quat_mul(_quat_mul(q_wxyz, qv), _quat_conj(q_wxyz))[1:]

    def transform_camera_quat(quat_wxyz):
        """Apply basis change for OpenCV->WebGL if requested."""
        if coord_mode == "da3":
            return quat_wxyz
        q_rot = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # 180deg about X
        return _quat_mul(q_rot, _quat_mul(quat_wxyz, q_rot))

    def _quat_from_axis180(axis):
        if axis == "x":
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        if axis == "y":
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        if axis == "z":
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _quat_from_axis90(axis):
        s = np.float32(np.sqrt(0.5))
        if axis == "x":
            return np.array([s, s, 0.0, 0.0], dtype=np.float32)
        if axis == "y":
            return np.array([s, 0.0, s, 0.0], dtype=np.float32)
        if axis == "z":
            return np.array([s, 0.0, 0.0, s], dtype=np.float32)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _parse_world_rot(spec):
        if spec == "none":
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        parts = [p.strip() for p in spec.split("+") if p.strip()]
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for part in parts:
            if part.endswith("90"):
                q_part = _quat_from_axis90(part[0])
            elif part.endswith("180"):
                q_part = _quat_from_axis180(part[0])
            else:
                q_part = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            q = _quat_mul(q_part, q)
        return q

    def apply_camera_extra_rot(quat_wxyz):
        q_extra = _parse_world_rot(camera_extra_rot)
        return _quat_mul(q_extra, quat_wxyz)

    def _quat_to_mat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)

    world_rot_q = _parse_world_rot(world_rot)
    world_rot_mat = _quat_to_mat(world_rot_q)
    world_shift_vec = np.array(world_shift, dtype=np.float32)

    point_dtype = np.float16 if use_fp16 else np.float32
    if use_fp16:
        print(f"  Using float16 for point cloud coordinates")

    for i, frame in enumerate(frames):
        # Get point cloud data
        pts = frame["points"].astype(np.float32)  # use float32 for transforms
        clrs = frame["colors"]

        # Apply downsampling if requested
        if downsample_indices is not None:
            pts = pts[downsample_indices]
            clrs = clrs[downsample_indices]

        # Transform points to glTF coordinate system
        pts_transformed = transform_opencv_to_gltf(pts) - center_offset
        pts_transformed = (world_rot_mat @ pts_transformed.T).T + world_shift_vec
        if scale != 1.0:
            pts_transformed *= scale

        # Ensure colors are in correct format (0-255 uint8)
        if isinstance(clrs, np.ndarray):
            if clrs.dtype == np.float64 or clrs.dtype == np.float32:
                if clrs.max() <= 1.0:
                    clrs_uint8 = (clrs * 255).astype(np.uint8)
                else:
                    clrs_uint8 = clrs.astype(np.uint8)
            else:
                clrs_uint8 = clrs.astype(np.uint8)
        else:
            clrs_uint8 = np.array(clrs, dtype=np.uint8)

        # Add/update point cloud
        server.scene.add_point_cloud(
            name="/video",
            points=pts_transformed.astype(point_dtype),
            colors=clrs_uint8,
            point_size=point_size,
        )

        # Add camera frustum - use float32
        cam_pos = frame["cam_pos"].astype(np.float32) - center_offset
        cam_pos = world_rot_mat @ cam_pos + world_shift_vec
        if scale != 1.0:
            cam_pos *= scale
        cam_quat = frame["cam_quat"].astype(np.float32)

        # Normalize quaternion defensively
        cam_quat = cam_quat / (np.linalg.norm(cam_quat) + 1e-12)

        # If input is world-to-camera, convert to camera-to-world first
        if camera_convention == "w2c":
            cam_quat = _quat_conj(cam_quat)
            cam_pos = -_rotate_vec(cam_quat, cam_pos)

        # Use DA3-aligned poses directly (already in viser coordinates)
        cam_pos_transformed = transform_camera_pos(cam_pos)
        cam_quat_transformed = transform_camera_quat(cam_quat)
        cam_quat_transformed = _quat_mul(world_rot_q, cam_quat_transformed)
        cam_quat_transformed = apply_camera_extra_rot(cam_quat_transformed)

        # Ensure proper tuple format
        cam_pos_tuple = tuple(cam_pos_transformed.tolist())
        cam_quat_tuple = tuple(cam_quat_transformed.tolist())

        # Handle image for camera frustum
        img = frame.get("image")
        if img is not None:
            if isinstance(img, np.ndarray):
                img = img.astype(np.uint8)

        server.scene.add_camera_frustum(
            name="/camera",
            fov=1.0,
            aspect=float(frame["aspect"]),
            scale=0.05,
            position=cam_pos_tuple,
            wxyz=cam_quat_tuple,
            image=img,
        )

        # Insert sleep for animation timing (except for last frame)
        if i < len(frames) - 1:
            serializer.insert_sleep(frame_duration)

        # Progress indicator
        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            print(f"    Processed {i + 1}/{len(frames)} frames")

    # Serialize and save
    print(f"  Serializing...")
    data = serializer.serialize()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Apply gzip compression if requested
    if use_gzip:
        print(f"  Compressing with gzip...")
        compressed_data = gzip.compress(data, compresslevel=6)
        compression_ratio = len(data) / len(compressed_data)
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        Path(output_path).write_bytes(compressed_data)
        file_size_mb = len(compressed_data) / 1024 / 1024
    else:
        Path(output_path).write_bytes(data)
        file_size_mb = len(data) / 1024 / 1024

    print(f"  Saved to {output_path} ({file_size_mb:.2f} MB)")

    # Stop server
    server.stop()
    return True


def export_project_dir(
    input_dir: str,
    output_dir: str,
    fps: float = 30.0,
    decay_filter: str = None,
    camera_convention: str = "c2w",
    camera_extra_rot: str = "none",
    center_mode: str = "none",
    world_rot: str = "none",
    coord_mode: str = "da3",
    world_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    use_gzip: bool = False,
    downsample: float = 1.0,
    use_fp16: bool = False,
    frame_step: int = 1,
    scale: float = 1.0,
):
    """
    Export all decay levels from a project directory.
    """
    # Find all decay subdirectories
    decay_dirs = sorted(glob.glob(os.path.join(input_dir, "decay_*")))

    if not decay_dirs:
        print(f"No decay_* directories found in {input_dir}")
        return

    print(f"Found {len(decay_dirs)} decay levels")

    os.makedirs(output_dir, exist_ok=True)

    # Compute shared center offset so all decay levels are aligned
    shared_center = None
    if center_mode == "all_mean":
        centers = []
        for dd in decay_dirs:
            pkl_p = os.path.join(dd, "viser_cache.pkl")
            if not os.path.exists(pkl_p):
                continue
            with open(pkl_p, 'rb') as f:
                frames = pickle.load(f)
            pts = frames[0]["points"].astype(np.float32)
            centers.append(pts.mean(axis=0))
            del frames
        if centers:
            shared_center = np.mean(centers, axis=0)
            print(f"Using shared center (mean of {len(centers)} decay levels): {shared_center}")
    elif center_mode != "none":
        ref_pkl = os.path.join(decay_dirs[0], "viser_cache.pkl")
        if os.path.exists(ref_pkl):
            with open(ref_pkl, 'rb') as f:
                ref_frames = pickle.load(f)
            ref_pts = ref_frames[0]["points"].astype(np.float32)
            if center_mode == "first_bbox":
                shared_center = (ref_pts.min(axis=0) + ref_pts.max(axis=0)) * 0.5
            elif center_mode == "first_mean":
                shared_center = ref_pts.mean(axis=0)
            print(f"Using shared center from {os.path.basename(decay_dirs[0])}: {shared_center}")
            del ref_frames

    for decay_dir in decay_dirs:
        pkl_path = os.path.join(decay_dir, "viser_cache.pkl")
        if not os.path.exists(pkl_path):
            print(f"  Skipping {decay_dir}: no viser_cache.pkl")
            continue

        # Extract decay value from directory name
        decay_name = os.path.basename(decay_dir)

        # Apply filter if specified
        if decay_filter:
            decay_val = decay_name.replace("decay_", "")
            if decay_val not in decay_filter.split(","):
                print(f"  Skipping {decay_name} (not in filter)")
                continue

        output_path = os.path.join(output_dir, f"{decay_name}.viser")

        # Skip if already exists
        if os.path.exists(output_path):
            print(f"  Skipping {decay_name}: already exists")
            continue

        try:
            export_single_pkl(
                pkl_path,
                output_path,
                fps=fps,
                camera_convention=camera_convention,
                camera_extra_rot=camera_extra_rot,
                center_mode=center_mode,
                world_rot=world_rot,
                coord_mode=coord_mode,
                world_shift=world_shift,
                use_gzip=use_gzip,
                downsample=downsample,
                use_fp16=use_fp16,
                frame_step=frame_step,
                scale=scale,
                center_override=shared_center,
            )
        except Exception as e:
            print(f"  Error exporting {decay_name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Export viser_cache.pkl to .viser format (OPTIMIZED)")
    parser.add_argument("--input", "-i", type=str, help="Input viser_cache.pkl file")
    parser.add_argument("--output", "-o", type=str, help="Output .viser file")
    parser.add_argument("--input-dir", type=str, help="Input project directory (exports all decay levels)")
    parser.add_argument("--output-dir", type=str, default="assets/viser", help="Output directory for .viser files")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback frame rate")
    parser.add_argument("--point-size", type=float, default=0.007, help="Point cloud point size")
    parser.add_argument(
        "--camera-convention",
        type=str,
        default="c2w",
        choices=["c2w", "w2c"],
        help="Camera pose convention stored in viser_cache.pkl: c2w (default) or w2c",
    )
    parser.add_argument(
        "--camera-extra-rot",
        type=str,
        default="none",
        help="Extra rotation applied to camera (e.g. 'y180', 'z90', or 'y180+z90')",
    )
    parser.add_argument(
        "--center",
        type=str,
        default="first_mean",
        choices=["none", "first_bbox", "first_mean", "all_mean"],
        help="Recenter scene: first_bbox/first_mean use decay_0.0 frame[0]; all_mean averages frame[0] mean across all decay levels",
    )
    parser.add_argument(
        "--world-rot",
        type=str,
        default="y180",
        help="Rotate the whole scene and camera (e.g. 'y180', 'z90', or 'y180+z90')",
    )
    parser.add_argument(
        "--coord-mode",
        type=str,
        default="da3",
        choices=["da3", "webgl"],
        help="Coordinate convention: da3 uses -Y up; webgl flips Y/Z to +Y up",
    )
    parser.add_argument(
        "--world-shift",
        type=str,
        default="0,0,0",
        help="Translate scene and camera by x,y,z (e.g. '0.1,-0.2,0')",
    )
    parser.add_argument("--decay-filter", type=str, help="Comma-separated decay values to export (e.g., '0.0,0.5,1.0')")
    parser.add_argument("--no-gzip", action="store_true", help="Disable gzip compression")
    parser.add_argument("--downsample", type=float, default=1.0, help="Fraction of points to keep (0.0-1.0), e.g. 0.5 = 50%%")
    parser.add_argument("--fp16", action="store_true", help="Use float16 for point cloud coordinates (smaller files)")
    parser.add_argument("--frame-step", type=int, default=1, help="Take every Nth frame (e.g. 2 = keep half the frames)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for scene (e.g. 0.125 to shrink 8x)")

    args = parser.parse_args()

    use_gzip = not args.no_gzip
    use_fp16 = args.fp16
    frame_step = args.frame_step

    if args.input:
        # Single file export
        output = args.output or args.input.replace(".pkl", ".viser")
        shift_vals = [float(v) for v in args.world_shift.split(",")]
        if len(shift_vals) != 3:
            raise ValueError("--world-shift must be three comma-separated numbers")
        export_single_pkl(
            args.input,
            output,
            fps=args.fps,
            point_size=args.point_size,
            camera_convention=args.camera_convention,
            camera_extra_rot=args.camera_extra_rot,
            center_mode=args.center,
            world_rot=args.world_rot,
            coord_mode=args.coord_mode,
            world_shift=tuple(shift_vals),
            use_gzip=use_gzip,
            downsample=args.downsample,
            use_fp16=use_fp16,
            frame_step=frame_step,
            scale=args.scale,
        )
    elif args.input_dir:
        # Directory export
        shift_vals = [float(v) for v in args.world_shift.split(",")]
        if len(shift_vals) != 3:
            raise ValueError("--world-shift must be three comma-separated numbers")
        export_project_dir(
            args.input_dir,
            args.output_dir,
            fps=args.fps,
            decay_filter=args.decay_filter,
            camera_convention=args.camera_convention,
            camera_extra_rot=args.camera_extra_rot,
            center_mode=args.center,
            world_rot=args.world_rot,
            coord_mode=args.coord_mode,
            world_shift=tuple(shift_vals),
            use_gzip=use_gzip,
            downsample=args.downsample,
            use_fp16=use_fp16,
            frame_step=frame_step,
            scale=args.scale,
        )
    else:
        parser.print_help()
        print("\nError: Must specify --input or --input-dir")


if __name__ == "__main__":
    main()
