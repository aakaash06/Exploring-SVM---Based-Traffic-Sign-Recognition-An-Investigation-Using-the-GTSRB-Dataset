# Exploring-SVM---Based-Traffic-Sign-Recognition-An-Investigation-Using-the-GTSRB-Dataset

This project explores traffic sign recognition using Support Vector Machines (SVM) and Convolutional Neural Networks (CNN), applying the German Traffic Sign Recognition Benchmark (GTSRB) dataset and testing on the German Traffic Sign Detection Benchmark (GTSDB) dataset to demonstrate real-world applicability.

## Project Structure
- **Project.ipynb**: Main notebook for SVM-based traffic sign recognition using the GTSRB dataset.
- **project vid.ipynb**: Notebook for testing the trained SVM model on the GTSDB dataset, to assess its performance in real-world scenarios.
- **DL part project.ipynb**: Implementation of a CNN model for comparison with SVM.
- **pythonproject/**: Contains the website implementation; run `server.py` to start the server.
- **saved_model.pkl** and **feature_selector.pkl**: Pre-trained SVM model and feature selector, saved for quick deployment in the website.

## Steps Followed in the Project

1. **Dataset Exploration**:
   - Analyzed GTSRB dataset structure and class distribution.

2. **Data Preprocessing**:
   - Resizing, normalization, and noise reduction (excluded after results analysis).

3. **Data Splitting**:
   - Employed stratified splitting for robust training and testing.

4. **Feature Extraction**:
   - Used Histogram of Oriented Gradients (HOG) and wavelet transforms.

5. **Feature Selection**:
   - Selected top 500 features (using Chi-squared method), which provided the best results.

6. **Model Training and Hyperparameter Tuning**:
   - Tuned hyperparameters (C, Kernel, Gamma, and Degree).
   - Optimal SVM model: C=1, Kernel=Linear, Gamma=Scale, achieving validation accuracy of 97.7% and test accuracy of 88.82%.
   - For CNN, optimal configuration: SGD optimizer, Sparse Categorical Cross Entropy as loss function, achieving train accuracy of 97.88% and test accuracy of 85.83%.

7. **Deployment**:
   - SVM model deployed on a webpage using JavaScript and HTML. For implementation, the saved model and feature selector files (`saved_model.pkl` and `feature_selector.pkl`) can be loaded directly.

8. **Testing on GTSDB**:
   - Tested model on GTSDB to verify if it can recognize traffic signs in entire scenes, validating its real-world use.

### Key Findings
SVM provided better test accuracy in this case, suggesting it can be more effective and cost-efficient for deployment compared to a deep learning model in similar scenarios.

## Running the Project

1. **Data**: Download the GTSRB dataset [here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and GTSDB dataset [here](https://benchmark.ini.rub.de/gtsdb_dataset.html).
2. **Server**: To run the web interface, navigate to the `pythonproject/` folder and run `server.py`.

## Acknowledgments
- [GTSRB dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- [GTSDB dataset from Benchmark](https://benchmark.ini.rub.de/gtsdb_dataset.html)

NOTE: The pkl file of the model is too big to upload, you can download it by running the project.ipynb
