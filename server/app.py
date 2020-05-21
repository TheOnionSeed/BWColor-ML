import os
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt

UPLOAD_FOLDER = './static/imgs'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def main():
   return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            
            
            
            resp = jsonify(success=True)
            return resp
if __name__ == "__main__":
   app.run(debug=True,host="localhost",port=5000)