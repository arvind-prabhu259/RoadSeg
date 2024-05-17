from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
from SegmentationModel.SegModel import load_model
import numpy as np
import asyncio
from process_img import process_image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'output/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


# UPLOAD_FOLDER = 'output/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


os.chdir("C:\\Users\\arvin\\MLProjectsFolder\\RoadSeg\\FlaskApp")
model = load_model('Unet_Model_Epoch_199.pth')

clahe = cv2.createCLAHE(clipLimit=40)

@app.route('/')
def home():
    return render_template('index.html', message = None)

@app.route('/return_home')
def return_home():
    return render_template('index.html', message = None)

@app.route('/upload', methods = ['POST'])
async def upload():
    if(request.method == 'POST'):
        file = request.files['file']
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # file.save(os.path.join(app.config['OUTPUT_FOLDER'], filename))
            # print(os.path.join(app.config['OUTPUT_FOLDER'], filename))
            
            processed_file_name = "processed_"+filename
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], processed_file_name)
            print(input_path, output_path)
            await process_image(input_path, output_path)

            return redirect(url_for('uploaded_file', filename = processed_file_name))
            
        return render_template('index.html', message = "Select an image")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print(filename)
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    # return render_template('upload.html', filename = filename)

@app.route('/output/<filename>')
def send_image(filename):
    # print(os.path.join(app.config['OUTPUT_FOLDER'], filename))
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)


# @app.route('/output/<filename>')
# def output_image(filename):
#     return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
#     return render_template('upload.html', output_image = filename)

# @app.route('/upload', methods = ['POST'])
# def upload():
#     if(request.method == 'POST'):
#         file = request.files['file']
#         if file:
#             filename = file.filename
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return f'<h1>Uploaded: {filename}</h1><img src="output/{filename}" alt="Output image">'
#         return "No file uploaded"