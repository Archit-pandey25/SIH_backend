from flask import Flask, render_template, request, jsonify
from ml_model import predict_image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # return jsonify({'error': 'No selected file'})
        # Save the uploaded image to a temporary folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Perform image classification using your pre-trained model
        result = predict_image(file_path)  # Implement this function in ml_model.py

        # You can also remove the uploaded file at this point
        os.remove(file_path)

        return jsonify({'result': result})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
