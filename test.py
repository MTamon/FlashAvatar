import os, sys 
import random
import numpy as np
import torch
import argparse
import cv2
import time
import datetime

from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='id1_25', help='id name')
    parser.add_argument('--logname', type=str, default='log', help='log name')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument("--checkpoint", type=str, default = None)
    parser.add_argument('--use-keep-list', dest='use_keep_list',
                        action='store_true',
                        help='Restrict the test video to frames in '
                             'dataset/<idname>/raw/keep_list.txt. By default '
                             'test.py renders every frame (including ones '
                             'excluded from training) so the video shows how '
                             'the model handles motion-blurred poses.')
    parser.add_argument('--driver_idname', type=str, default=None,
                        help='Drive the trained avatar with another identity\'s '
                             'preprocessed FLAME features '
                             '(metrical-tracker/output/<driver_idname>/). The '
                             'trained person\'s shape (and checkpointed '
                             'Gaussians / MLP) is kept, while per-frame '
                             'expression, jaw, eye, eyelid, and camera (R, T, K) '
                             'are taken from the driver. This enables '
                             'cross-identity reenactment without retraining, '
                             'since FlashAvatar conditions on identity shape but '
                             'does NOT bake identity into the driving signal.')
    parser.add_argument('--driver_range', type=str, default=None,
                        help='Driver frame range as "start:end" (0-indexed, '
                             'end-exclusive). Defaults to the full driver '
                             'sequence when --driver_idname is set.')
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    batch_size = 1
    set_random_seed(args.seed)

    ## deform model
    DeformModel = Deform_Model(args.device).to(args.device)
    DeformModel.training_setup()
    DeformModel.eval()

    ## dataloader
    data_dir = os.path.join('dataset', args.idname)
    mica_datadir = os.path.join('metrical-tracker/output', args.idname)
    logdir = data_dir+'/'+args.logname

    driver_mica_datadir = None
    driver_datadir = None
    driver_range = None
    if args.driver_idname is not None:
        driver_mica_datadir = os.path.join('metrical-tracker/output',
                                           args.driver_idname)
        driver_datadir = os.path.join('dataset', args.driver_idname)
        if not os.path.isdir(os.path.join(driver_mica_datadir, 'checkpoint')):
            raise FileNotFoundError(
                f"driver tracker output not found: "
                f"{driver_mica_datadir}/checkpoint. Pre-process the driver "
                f"video (`scripts/preprocess.py prepare` + "
                f"`scripts/run_tracker.sh <driver_idname>`) first.")
        if not os.path.isdir(os.path.join(driver_datadir, 'imgs')):
            # RGB frames are only used for side-by-side display; missing
            # imgs are non-fatal (the left half of the canvas will be black).
            driver_datadir = None
        if args.driver_range is not None:
            parts = args.driver_range.split(':')
            if len(parts) != 2:
                raise ValueError(
                    f"--driver_range must be formatted as 'start:end', got "
                    f"{args.driver_range!r}")
            driver_range = (int(parts[0]), int(parts[1]))

    scene = Scene_mica(data_dir, mica_datadir, train_type=1,
                       white_background=lpt.white_background, device=args.device,
                       use_keep_list=args.use_keep_list,
                       driver_mica_datadir=driver_mica_datadir,
                       driver_datadir=driver_datadir,
                       driver_range=driver_range)
    
    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)

    if args.checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.checkpoint, weights_only=False)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    viewpoint = scene.getCameras().copy()
    if not viewpoint:
        raise RuntimeError("scene produced 0 cameras; nothing to render.")
    # Use the per-camera resolution (driver and identity may differ).
    canvas_h = int(viewpoint[0].image_height)
    canvas_w = int(viewpoint[0].image_width)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if args.driver_idname is not None:
        os.makedirs(logdir, exist_ok=True)
        vid_name = f'test_driven_by_{args.driver_idname}.avi'
    else:
        vid_name = 'test.avi'
    vid_save_path = os.path.join(logdir, vid_name)
    out = cv2.VideoWriter(vid_save_path, fourcc, 25, (canvas_w*2, canvas_h), True)

    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    for iteration in range(len(viewpoint)):
        viewpoint_cam = viewpoint[iteration]
        frame_id = viewpoint_cam.uid

        # deform gaussians
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['eyes_pose'] = viewpoint_cam.eyes_pose
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose
        codedict['head_pose'] = viewpoint_cam.head_pose
        verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        # Render
        render_pkg = render(viewpoint_cam, gaussians, ppt, background)
        image= render_pkg["render"]
        image = image.clamp(0, 1)

        gt_image = viewpoint_cam.original_image
        save_image = np.zeros((canvas_h, canvas_w*2, 3))
        gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy()
        image_np = (image*255.).permute(1,2,0).detach().cpu().numpy()

        save_image[:, :canvas_w, :] = gt_image_np
        save_image[:, canvas_w:, :] = image_np
        save_image = save_image.astype(np.uint8)
        save_image = save_image[:,:,[2,1,0]]

        out.write(save_image)
    out.release()
    print(f"[test] wrote {vid_save_path}")
    
    
   
        

           