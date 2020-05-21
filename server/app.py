import os
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import io
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import base64
from PIL import Image

UPLOAD_FOLDER = './static/imgs'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# NN models
prototxt = "./model/colorization_deploy_v2.prototxt"
model = "./model/colorization_release_v2.caffemodel"
points = "./model/pts_in_hull.npy"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def main():
   return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convImg(image):
   net = cv2.dnn.readNetFromCaffe(prototxt, model)
   pts = np.load(points)

   class8 = net.getLayerId("class8_ab")
   conv8 = net.getLayerId("conv8_313_rh")
   pts = pts.transpose().reshape(2, 313, 1, 1)
   net.getLayer(class8).blobs = [pts.astype("float32")]
   net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

   # Get the lightness 
   scaled = image.astype("float32") / 255.0
   lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
   resized = cv2.resize(lab, (224, 224))
   L = cv2.split(resized)[0]
   L -= 50
   
   # Predicting ab
   # a: green–red
   # b: blue-yellow
   net.setInput(cv2.dnn.blobFromImage(L))
   ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
   ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
   L = cv2.split(lab)[0]
   colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
   colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
   colorized = np.clip(colorized, 0, 1)
   colorized = (255 * colorized).astype("uint8")
   
   #plt.imshow(colorized)
   #plt.axis('off');
   #plt.show()

   return colorized

@app.route('/uploader', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'imagefile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['imagefile']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #file.save(app.config['UPLOAD_FOLDER']+"/"+ filename)
            
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            
            img_data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
            color_image_flag = 1
            img = cv2.imdecode(img_data, color_image_flag)
            
            # Convert grayscale img to colored
               #plt.imshow(colorized)
               
            colImg = convImg(img)
            out_memory_file = io.BytesIO()
            plt.imsave(out_memory_file, colImg)
            
           
            # Return converted image
            encoded_img = base64.encodebytes(out_memory_file.getvalue()).decode('ascii')
            
            response =  { 'Status' : 'Success', 'Image': encoded_img}
            resp = jsonify(response)
            return resp
if __name__ == "__main__":
   app.run(debug=True,host="localhost",port=5000)