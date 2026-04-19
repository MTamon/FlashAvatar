"""Face parsing via BiSeNet trained on CelebAMask-HQ.

For each input frame this module emits two binary masks:
  - XXXXX_neckhead.png : head + neck + hair silhouette
  - XXXXX_mouth.png    : inner-mouth region

BiSeNet was trained at 512x512. Frames of arbitrary resolution are resized
down to 512x512 for inference and the resulting label map is upsampled back
to the original resolution with nearest-neighbour, so masks stay in the same
coordinate system as the input images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .models.bisenet import BiSeNet
from .download import ensure_bisenet_weights

# CelebAMask-HQ class indices (see models/bisenet.py header).
HEAD_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18}
MOUTH_CLASSES = {11}

# ImageNet mean / std as used by face-parsing.PyTorch.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FaceParser:
    def __init__(self, weights_path: Path | None = None,
                 device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        self.net = BiSeNet(n_classes=19).to(self.device).eval()
        ckpt = ensure_bisenet_weights(weights_path.parent if weights_path else None)
        if weights_path is not None and Path(weights_path).is_file():
            ckpt = Path(weights_path)
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.net.load_state_dict(state)
        self._mean = _MEAN.to(self.device)
        self._std = _STD.to(self.device)

    @torch.no_grad()
    def predict_labels(self, img: Image.Image) -> np.ndarray:
        """Return a HxW int64 array of class labels at the image's native size."""
        w, h = img.size
        x = torch.from_numpy(np.asarray(img.convert("RGB"))).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        logits = self.net(x)[0]
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)[0].to(torch.int64).cpu().numpy()


def _mask_from_labels(labels: np.ndarray, class_ids: set[int]) -> np.ndarray:
    out = np.zeros(labels.shape, dtype=np.uint8)
    for c in class_ids:
        out[labels == c] = 255
    return out


def run_parsing(imgs_dir: Path, out_dir: Path, weights_path: Path | None = None,
                device: str = "cuda", overwrite: bool = False) -> int:
    """Generate neckhead / mouth masks for every frame in `imgs_dir`.

    Returns the number of frames processed.
    """
    imgs_dir = Path(imgs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(imgs_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no *.jpg frames under {imgs_dir}")

    parser = FaceParser(weights_path=weights_path, device=device)

    done = 0
    for frame in tqdm(frames, desc="parsing"):
        stem = frame.stem
        head_path = out_dir / f"{stem}_neckhead.png"
        mouth_path = out_dir / f"{stem}_mouth.png"
        if not overwrite and head_path.exists() and mouth_path.exists():
            done += 1
            continue
        labels = parser.predict_labels(Image.open(frame))
        Image.fromarray(_mask_from_labels(labels, HEAD_CLASSES), mode="L").save(head_path)
        Image.fromarray(_mask_from_labels(labels, MOUTH_CLASSES), mode="L").save(mouth_path)
        done += 1
    return done
