
import matplotlib.pylab as plt
import numpy as np
from skimage.transform import resize
from flask_cors import CORS
import pickle
from os import path as os_path
from flask import Flask, request, jsonify, render_template


default_image_size = tuple((256, 256))
UPLOAD_FOLDER = 'uploads/'
categories = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]


app = Flask(__name__,static_url_path="/static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app, resources={r"/train": {"origins": "*"}})
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def getPrediction():
    img = plt.imread('uploads/test.jpg')
    resImage = resize(img, default_image_size)
    
    model_path = os_path.join(app.static_folder, 'cnn_model.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    resImage = resImage.astype(np.float32) / 255.0
    prob = model.predict(np.array([resImage]))
    predicted_class_index = np.argmax(prob[0])
    predicted_class_label = categories[predicted_class_index % len(categories)] 
    
    return predicted_class_label


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def home():
    return render_template('upload.html')


@app.route('/api/file-upload', methods=['GET', 'POST'])
def upload_file():
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400

        return resp
    file = request.files['file']
    
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400

        return resp
        
    if file and allowed_file(file.filename):
        filename = "test.jpg"
        file.save(os_path.join(app.config['UPLOAD_FOLDER'], filename))

        return render_template('answer.html', answer=getPrediction())
    else:
        resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400

        return resp

if __name__ == "__main__":
    app.run()

