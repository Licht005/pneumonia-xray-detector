# Grad-CAM visualization logic
import torch
import numpy as np
import cv2
from torchvision import transforms
from gradcam import GradCAM
from pytorch_grad_cam import visualize_cam



def get_gradcam_heatmap(model, image_tensor, target_class=None):
    model.eval()
    cam_extractor = GradCAM(model, target_layer=model.layer4[-1])

    mask, _ = cam_extractor(image_tensor, class_idx=target_class)
    heatmap, result = visualize_cam(mask, image_tensor)

    return heatmap, result
