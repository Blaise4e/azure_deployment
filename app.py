import pickle as pc
import os
from flask import Flask, flash, request, redirect, url_for, render_template, after_this_request
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath
import torch
from app_function import MedNet, predict_img
from werkzeug.utils import secure_filename

app = Flask(__name__)


# Config options - Make sure you created a 'config.py' file.
#app.config.from_object('config')
# To get one variable, tape app.config['MY_VARIABLE']

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template("index.html")
 
model = torch.load('saved_model', map_location=torch.device('cpu'))

@app.route('/result', methods = ['GET', 'POST'])
def result():  
    
    upload_path = join(dirname(realpath(__file__)), "static/img")
    IMAGE_UPLOADS = upload_path
    app.config["IMAGE_UPLOADS"] = upload_path
    if request.method == "POST":

        if request.files:
            
            image = request.files["image"]
            image.save(os.path.join(app.config['IMAGE_UPLOADS'], image.filename))
            #full_filename = os.path.join(app.config['IMAGE_UPLOADS'], image.filename)
            filename = secure_filename(image.filename)
            pred = predict_img(image, model)
            #os.remove(full_filename)
               

    return render_template("result.html", pred= 'Ceci est : {}'.format(pred), img_filename= 'static/img/' + filename)
     
"""
                @after_this_request 
                def remove_file(response):
                os.remove(full_filename)
                return response            
"""
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
    
