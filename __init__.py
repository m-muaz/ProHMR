import argparse
import os
from typing import Union

import cv2
import torch
from prohmr.configs import dataset_config, get_config, prohmr_config
from prohmr.datasets import OpenPoseDataset
from prohmr.models import ProHMR
from prohmr.optimization import KeypointFitting
from prohmr.utils import recursive_to
from prohmr.utils.renderer import Rendere
from tqdm import tqdm


def initialize_3d_model(checkpoint_path, 
                        model_cfg=None,
                        img_folder=None,
                        keypoint_folder=None,
                        run_fitting=False,
                        full_frame=False,
                        batch_size=1
                        ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if model_cfg:
        model_cfg = get_config(model_cfg)
    else:
        model_cfg = prohmr_config()
        
    # Setup model (ProHMR model)
    model = ProHMR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg).to(device)
    model.eval()
    
    if run_fitting:
        keypoint_fitting = KeypointFitting(model_cfg)
        
    # Create dataset on-the-fly
    dataset = OpenPoseDataset(model_cfg, img_folder=img_folder, keypoint_folder=keypoint_folder, max_people_per_image=1)
    
    # Setup dataloader with batch_size = 1 (Process images sequentially)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    
    return model, dataloader

def render_image(predicted_params, img):
    """
    Renders 3D pose and shape onto input image
    """
    pass
            
        