import os, sys
import random
import json
from PIL import Image
import torch
import math
import numpy as np
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov


class Scene_mica:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device,
                 use_keep_list=True,
                 driver_mica_datadir=None, driver_datadir=None, driver_range=None):
        ## train_type: 0 for train, 1 for test, 2 for eval
        #
        # Cross-identity driving (test-time reenactment):
        #   When `driver_mica_datadir` is given, `self.shape_param` is still taken
        #   from the trained identity (so Gaussians and the DeformModel's
        #   canonical UV encoding stay consistent), but per-frame FLAME
        #   parameters (exp / jaw / eyes / eyelids), the camera pose (R, T)
        #   and intrinsics (K) are read from the driver's metrical-tracker
        #   output. `driver_datadir` is only used to locate RGB frames for
        #   side-by-side display; parsing/alpha are NOT required.
        #   `driver_range=(start, end)` restricts the driver frames; default
        #   is the full driver sequence.
        frame_delta = 1 # default mica-tracking starts from the second frame
        driving = driver_mica_datadir is not None

        # Always take the identity shape from the trained person, even under
        # driving: Gaussians / MLP weights in the checkpoint are tied to it.
        identity_ckpt_path = os.path.join(mica_datadir, 'checkpoint', '00000.frame')
        identity_payload = torch.load(identity_ckpt_path, weights_only=False)
        self.shape_param = torch.as_tensor(identity_payload['flame']['shape'])

        if driving:
            # Per-frame FLAME + camera come from the driver.
            source_mica_datadir = driver_mica_datadir
            images_folder = (os.path.join(driver_datadir, "imgs")
                             if driver_datadir else None)
            parsing_folder = None
            alpha_folder = None
            composite_gt = False
        else:
            source_mica_datadir = mica_datadir
            images_folder = os.path.join(datadir, "imgs")
            parsing_folder = os.path.join(datadir, "parsing")
            alpha_folder = os.path.join(datadir, "alpha")
            composite_gt = True

        # Optional motion-blur filter: skip frames whose image stems are not
        # listed in raw/keep_list.txt. The file is written by
        # `preprocess filter-blur`; when absent (or use_keep_list=False) all
        # frames are used. keep_list is identity-specific and therefore
        # ignored during cross-identity driving.
        keep_set = None
        if use_keep_list and not driving:
            keep_list_path = os.path.join(datadir, "raw", "keep_list.txt")
            if os.path.isfile(keep_list_path):
                with open(keep_list_path) as fh:
                    keep_set = {ln.strip() for ln in fh if ln.strip()}
                print(f"[scene] using keep_list with {len(keep_set)} frames: "
                      f"{keep_list_path}")

        mica_ckpt_dir = os.path.join(source_mica_datadir, 'checkpoint')
        self.N_frames = len(os.listdir(mica_ckpt_dir))
        self.cameras = []
        test_num = 500
        eval_num = 50
        max_train_num = 10000
        train_num = min(max_train_num, self.N_frames - test_num)
        ckpt_path = os.path.join(mica_ckpt_dir, '00000.frame')
        payload = torch.load(ckpt_path, weights_only=False)
        orig_w, orig_h = payload['img_size']
        K = payload['opencv']['K'][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        FovY = focal2fov(fl_y, orig_h)
        FovX = focal2fov(fl_x, orig_w)

        self.bg_image = torch.zeros((3, int(orig_h), int(orig_w)))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        if driving:
            if driver_range is not None:
                range_down, range_up = driver_range
                range_down = max(0, int(range_down))
                range_up = min(self.N_frames, int(range_up))
            else:
                range_down, range_up = 0, self.N_frames
            print(f"[scene] driving mode: identity shape from "
                  f"{mica_datadir}, per-frame FLAME/camera from "
                  f"{driver_mica_datadir}, frames [{range_down}, {range_up})")
        else:
            if train_type == 0:
                range_down = 0
                range_up = train_num
            if train_type == 1:
                range_down = self.N_frames - test_num
                range_up = self.N_frames
            if train_type == 2:
                range_down = self.N_frames - eval_num
                range_up = self.N_frames

        skipped = 0
        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5) # obey mica tracking
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            if keep_set is not None and image_name_ori not in keep_set:
                skipped += 1
                continue
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'.frame')
            payload = torch.load(ckpt_path, weights_only=False)

            flame_params = payload['flame']
            exp_param = torch.as_tensor(flame_params['exp'])
            eyes_pose = torch.as_tensor(flame_params['eyes'])
            eyelids = torch.as_tensor(flame_params['eyelids'])
            jaw_pose = torch.as_tensor(flame_params['jaw'])

            oepncv = payload['opencv']
            w2cR = oepncv['R'][0]
            w2cT = oepncv['t'][0]
            R = np.transpose(w2cR) # R is stored transposed due to 'glm' in CUDA code
            T = w2cT

            image_path = (os.path.join(images_folder, image_name_ori+'.jpg')
                          if images_folder else None)
            if image_path and os.path.isfile(image_path):
                image = Image.open(image_path)
                # In driving mode the driver frame may not be exactly the
                # tracker's img_size (e.g. unfinalized preprocessing). Resize
                # so the side-by-side canvas and the rendered image share one
                # resolution.
                if driving and image.size != (int(orig_w), int(orig_h)):
                    image = image.resize((int(orig_w), int(orig_h)))
                resized_image_rgb = PILtoTensor(image)
                gt_image = resized_image_rgb[:3, ...]
            else:
                gt_image = torch.zeros(3, int(orig_h), int(orig_w))

            if composite_gt:
                # alpha
                alpha_path = os.path.join(alpha_folder, image_name_ori+'.jpg')
                alpha = Image.open(alpha_path)
                alpha = PILtoTensor(alpha)

                # # if add head mask
                head_mask_path = os.path.join(parsing_folder, image_name_ori+'_neckhead.png')
                head_mask = Image.open(head_mask_path)
                head_mask = PILtoTensor(head_mask)
                gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
                gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

                # mouth mask
                mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png')
                mouth_mask = Image.open(mouth_mask_path)
                mouth_mask = PILtoTensor(mouth_mask)
            else:
                # Driving mode: masks are unused downstream (test.py only
                # reads `original_image`), so dummies suffice.
                head_mask = torch.ones(1, gt_image.shape[1], gt_image.shape[2])
                mouth_mask = torch.ones(1, gt_image.shape[1], gt_image.shape[2])

            camera_indiv = Camera(colmap_id=frame_id, R=R, T=T,
                                FoVx=FovX, FoVy=FovY,
                                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
                                exp_param=exp_param, eyes_pose=eyes_pose, eyelids=eyelids, jaw_pose=jaw_pose,
                                image_name=image_name_mica, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)

        if keep_set is not None:
            split = {0: "train", 1: "test", 2: "eval"}.get(train_type, str(train_type))
            print(f"[scene] split={split}: {len(self.cameras)} kept / "
                  f"{skipped} skipped in range [{range_down}, {range_up})")
            if not self.cameras:
                raise RuntimeError(
                    f"no frames remain after keep_list filtering for "
                    f"split={split}; lower --percentile in `preprocess "
                    f"filter-blur`, or pass use_keep_list=False.")

    def getCameras(self):
        return self.cameras





    
