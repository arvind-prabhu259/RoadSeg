import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
from SegmentationModel.SegModel import load_model
import matplotlib.pyplot as plt
import pandas as pd

model = load_model('Unet_Model_Epoch_199.pth')

clahe = cv2.createCLAHE(clipLimit=40)

df = pd.read_csv('static\labels.csv')#, index_col='LabelNo')
color_dict = {row['LabelNo']: (row['r'], row['g'], row['b']) for _, row in df.iterrows()}

async def process_image(input_file_path, output_file_path):
    print(input_file_path, output_file_path)
    image = cv2.imread(input_file_path)

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (960, 720))

    clahe_img = clahe.apply(gray_img)
    
    torch_img = torch.from_numpy(clahe_img).float()
    torch_img = torch_img.unsqueeze(dim = 0).unsqueeze(dim = 0)

    model.eval()
    with torch.inference_mode():
        mask_preds = model(torch_img)
    mask_preds = mask_preds.permute((2,3,1,0)).numpy()

    mask_pred = np.argmax(mask_preds, axis=2)

    mask_pred = np.squeeze(mask_pred, axis=-1)
    mask_pred = mask_pred.astype(np.uint8)

    mask_rgb = np.zeros((mask_pred.shape[0], mask_pred.shape[1], 3))
    
    for label, color in color_dict.items():
        mask_rgb[mask_pred == label] = color

    mask_rgb = mask_rgb.astype(np.uint8)
    
    save_image = Image.fromarray(mask_rgb) # Saving as grayscale image 
    save_image.save(output_file_path)