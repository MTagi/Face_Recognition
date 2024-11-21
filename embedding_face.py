#pytorch
import torch
from PIL import Image
from torchvision import transforms
import torchvision

#other lib
import sys
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from detect import detect_face
# model_embedded_face (insightface)
from iresnet import iresnet100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_emb = insight_face(path="insightface/ckpt_epoch_50.pth", device=device, train=True)
weight = torch.load("E:/3.hocki1nam4/xu li anh/face_recognition/weights/arcface_r100.pth", map_location = device)
model_emb = iresnet100()
model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()


@torch.no_grad()
def get_feature(face_image):
    """
    Extract features from a face image.

    Args:
        face_image: The input face image.

    Returns:
        numpy.ndarray: The extracted features.
    """
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Preprocess image (BGR)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # Inference to get feature
        emb_img_face = model_emb(face_image).cpu().numpy()

    # Convert to array
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)

    return images_emb

