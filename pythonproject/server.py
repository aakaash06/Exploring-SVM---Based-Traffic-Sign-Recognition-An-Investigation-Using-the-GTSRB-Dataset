from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from util import load_model, process_image, select_features, predict_class, get_class_name

app = Flask(__name__)

model, selector = load_model(r"C:\Users\aakaash\OneDrive\Desktop\QMUL\Project\PROJECT\archive\artefacts\pythonProject\saved_model.pkl", r"C:\Users\aakaash\OneDrive\Desktop\QMUL\Project\PROJECT\archive\artefacts\pythonProject\feature_selector.pkl") #load the model

#serve html file
@app.route('/')
def index():
    return render_template('app.html')

#serve static files like js
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        temp_path = os.path.join('temp', file.filename) #uploaded img to temp file
        file.save(temp_path)
        image_features = process_image(temp_path) #preprocess
        selected_features = select_features(selector, image_features) #feature selection
        class_id = predict_class(model, selected_features)
        class_name = get_class_name(class_id) #get class name
        os.remove(temp_path)
        return jsonify({'class_id': int(class_id), 'class_name': class_name})


if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')

    app.run(debug=True)
