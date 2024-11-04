import joblib
import cv2
import numpy as np
import pywt
from skimage.feature import hog

CLASS_NAMES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

def get_class_name(class_id):
    return CLASS_NAMES.get(class_id, "Unknown class")

def load_model(model_path, selector_path): #to load the model
    model = joblib.load(model_path)
    selector = joblib.load(selector_path)
    return model, selector

#image preprocessing
def process_image(image_path):
    img = cv2.imread(image_path) #read
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32)) #resize
    img = img / 255.0

    #feature extraction
    hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    wavelet_features = np.hstack((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))

    combined_features = np.hstack((hog_features, wavelet_features))
    combined_features = np.maximum(combined_features, 0)  # Ensure non-negative

    return combined_features.reshape(1, -1)

#feature selection
def select_features(selector, image_features):
    selected_features = selector.transform(image_features)
    return selected_features

def predict_class(model, image_features):
    prediction = model.predict(image_features)
    return prediction[0]