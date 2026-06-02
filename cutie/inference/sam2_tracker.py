import os
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

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
            from sam2.build_sam import build_sam2_video_predictor
            import sam2 as _sam2_pkg

            # Resolve config: accept full path OR bare name
            if os.path.isfile(config):
                # Full path given (e.g. from Kaggle) — copy into sam2 configs dir
                import shutil as _shutil
                sam2_configs_dir = os.path.join(os.path.dirname(_sam2_pkg.__file__), 'configs')
                dest = os.path.join(sam2_configs_dir, os.path.basename(config))
                if not os.path.exists(dest):
                    _shutil.copy2(config, dest)
                config_name = os.path.splitext(os.path.basename(config))[0]  # strip .yaml
            else:
                # Bare name given — strip .yaml if present
                config_name = os.path.splitext(os.path.basename(config))[0]

            print(f"Loading SAM2 model from checkpoint: {checkpoint} with config: {config_name}...")
            self.predictor = build_sam2_video_predictor(config_name, checkpoint, device=device)
            self.config_path = config
            self.checkpoint_path = checkpoint


        # Reset previous inference state
        self.reset()

        # Handle different input formats
        if isinstance(video_path_or_frames, str) and os.path.isdir(video_path_or_frames):
            self.video_path = video_path_or_frames
        else:
            # Create a temporary directory to store frames as images
            self.temp_dir = tempfile.mkdtemp()
            self.video_path = self.temp_dir
            for idx, frame in enumerate(video_path_or_frames):
                if isinstance(frame, np.ndarray):
                    img = Image.fromarray(frame)
                elif isinstance(frame, Image.Image):
                    img = frame
                else:
                    # Assume PyTorch Tensor
                    img = Image.fromarray((frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
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
