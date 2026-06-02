import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil


def _ensure_sam2_importable():
    """
    Ensure sam2 is importable by searching for build_sam.py and fixing sys.path.
    Also clears stale namespace-package cache entries for 'sam2'.
    Returns the path to the sam2 package directory (containing build_sam.py).
    """
    # Known candidate roots (Kaggle, local, etc.)
    search_roots = ['/kaggle/working', '/kaggle', os.path.expanduser('~')]
    # Also add any directory already in sys.path that might contain sam2
    for p in sys.path:
        if p and p not in search_roots:
            search_roots.append(p)

    build_sam_path = None
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, files in os.walk(root):
            # Skip heavy directories to keep search fast
            dirnames[:] = [d for d in dirnames if d not in ('__pycache__', '.git', 'node_modules')]
            if 'build_sam.py' in files and 'sam2_video_predictor.py' in files:
                build_sam_path = os.path.join(dirpath, 'build_sam.py')
                break
        if build_sam_path:
            break

    if build_sam_path is None:
        raise ImportError(
            "Could not find sam2/build_sam.py. "
            "Make sure SAM2 is cloned to /kaggle/working/sam2 or is pip-installed."
        )

    sam2_pkg_dir = os.path.dirname(build_sam_path)      # .../sam2/sam2/
    sam2_install_root = os.path.dirname(sam2_pkg_dir)   # .../sam2/

    # Fix sys.path
    if sam2_install_root not in sys.path:
        sys.path.insert(0, sam2_install_root)
        print(f"  [SAM2] Added to sys.path: {sam2_install_root}")

    # Clear stale sam2 namespace-package cache (no __file__)
    stale = [k for k, v in sys.modules.items()
             if 'sam2' in k and getattr(v, '__file__', 'x') is None]
    for k in stale:
        del sys.modules[k]
    if stale:
        print(f"  [SAM2] Cleared stale modules: {stale}")

    return sam2_pkg_dir   # .../sam2/sam2/


def _resolve_config_name(config, sam2_pkg_dir):
    """
    Resolve the Hydra config_name from a full yaml path or bare name.
    SAM2 2.1 stores configs in configs/sam2.1/ subdir, so Hydra needs
    'sam2.1/sam2.1_hiera_l', not just 'sam2.1_hiera_l'.
    """
    configs_root = os.path.join(sam2_pkg_dir, 'configs')
    basename = os.path.basename(config)          # sam2.1_hiera_l.yaml
    name_no_ext = os.path.splitext(basename)[0]  # sam2.1_hiera_l

    print(f"  [SAM2] configs_root: {configs_root}")

    if os.path.isdir(configs_root):
        for dirpath, _, files in os.walk(configs_root):
            for f in files:
                if os.path.splitext(f)[0] == name_no_ext:
                    rel = os.path.relpath(os.path.join(dirpath, f), configs_root)
                    result = os.path.splitext(rel)[0].replace('\\', '/')
                    print(f"  [SAM2] Resolved config: {result}")
                    return result

    print(f"  [SAM2] Fallback config name: {name_no_ext}")
    return name_no_ext


class SAM2Tracker:
    def __init__(self):
        self.predictor = None
        self.inference_state = None
        self.temp_dir = None
        self.device = 'cuda'
        self.config_path = None
        self.checkpoint_path = None
        self.video_path = None

    def initialize(self, video_path_or_frames, checkpoint, config, device='cuda'):
        self.device = device

        # Load predictor if not already loaded or if checkpoint/config changed
        if self.predictor is None or self.config_path != config or self.checkpoint_path != checkpoint:
            # Step 1: Make sure sam2 is importable
            sam2_pkg_dir = _ensure_sam2_importable()

            # Step 2: Resolve config name for Hydra
            config_name = _resolve_config_name(config, sam2_pkg_dir)

            # Step 3: Build predictor
            from sam2.build_sam import build_sam2_video_predictor
            print(f"  Loading SAM2 from: {checkpoint}")
            print(f"  Config: {config_name}")
            self.predictor = build_sam2_video_predictor(config_name, checkpoint, device=device)
            self.config_path = config
            self.checkpoint_path = checkpoint

        self.reset()

        # Handle different input formats
        if isinstance(video_path_or_frames, str) and os.path.isdir(video_path_or_frames):
            self.video_path = video_path_or_frames
        else:
            self.temp_dir = tempfile.mkdtemp()
            self.video_path = self.temp_dir
            for idx, frame in enumerate(video_path_or_frames):
                if isinstance(frame, np.ndarray):
                    img = Image.fromarray(frame)
                elif isinstance(frame, Image.Image):
                    img = frame
                else:
                    img = Image.fromarray(
                        (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    )
                img.save(os.path.join(self.temp_dir, f"{idx:05d}.jpg"))

        # Initialize SAM2 inference state
        self.inference_state = self.predictor.init_state(video_path=self.video_path)

    def add_mask(self, frame_idx, mask, obj_id=1):
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).to(device=self.device)
        else:
            mask_tensor = mask.to(device=self.device)

        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.squeeze(0)

        self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask_tensor
        )

    def propagate(self, start_frame_idx, reverse=False):
        return self.predictor.propagate_in_video(
            inference_state=self.inference_state,
            start_frame_idx=start_frame_idx,
            reverse=reverse
        )

    def reset(self):
        if self.predictor is not None and self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
            self.inference_state = None
        if self.temp_dir is not None and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up temp directory {self.temp_dir}: {e}")
            self.temp_dir = None
