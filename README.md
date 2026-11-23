# Task 4 - Image Classification with Classical / Pre-trained ML

## Objective
Build an image-classification pipeline using the MNIST benchmark dataset. The project loads images, normalizes pixel values, trains a lightweight CNN, evaluates all digit classes, and demonstrates predictions with confidence scores.

## Technology
- Python
- TensorFlow / Keras (open-source)
- NumPy
- Scikit-learn (evaluation metrics)
- Matplotlib (charts and sample images)

## Dataset
MNIST contains 70,000 grayscale handwritten-digit images (28 x 28 pixels):
- 60,000 training images
- 10,000 test images
- 10 classes: digits 0 through 9

The dataset is downloaded automatically by TensorFlow the first time `train.py` runs. An internet connection is required for this first download. After it is cached locally, training and inference run locally.

## Architecture choice
A small Convolutional Neural Network (CNN) is used because convolution and pooling layers learn useful visual patterns in image data.

| Layer | Configuration |
|---|---|
| Input | 28 x 28 x 1 grayscale image |
| Convolution 1 | 32 filters, 3 x 3, ReLU |
| Max pooling 1 | 2 x 2 |
| Convolution 2 | 64 filters, 3 x 3, ReLU |
| Max pooling 2 | 2 x 2 |
| Flatten | Converts feature maps to a vector |
| Dense | 128 units, ReLU |
| Dropout | 0.3 |
| Output | 10 units, Softmax |

## Hyperparameters
- Epochs: 5
- Batch size: 128
- Optimizer: Adam
- Learning rate: 0.001
- Loss: Sparse categorical cross-entropy
- Validation split: 10%
- Random seed: 42

## Setup and execution (VS Code / terminal)

1. Extract the ZIP file.
2. Open the extracted `Task4_Image_Classification` folder in VS Code.
3. Open Terminal -> New Terminal.
4. Create a virtual environment (recommended):

   Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   macOS / Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install packages:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Train and evaluate:
   ```bash
   python train.py
   ```

7. Run a prediction after training:
   ```bash
   python predict.py
   ```
   To select another test image:
   ```bash
   python predict.py --index 25
   ```

Use Python 3.10 or 3.11 for the smoothest TensorFlow installation. Training is CPU-compatible; a GPU is not required.

## Deliverables generated in `outputs/`
- `mnist_cnn.keras` - trained model
- `training_curves.png` - training and validation loss/accuracy curves
- `confusion_matrix.png` - confusion matrix for all ten classes
- `sample_predictions.png` - sample test images with actual label, predicted label and confidence
- `prediction_logs.csv` - sample prediction log with ground truth and confidence
- `classification_report.txt` - per-class precision, recall and F1-score
- `metrics.txt` - test loss and accuracy

The output files are created after running `python train.py`. The first run may take longer because the MNIST dataset and Python packages need to be downloaded.

## Requirement checklist
- [x] Benchmark image dataset (MNIST)
- [x] Image loading and pixel normalization
- [x] Train-test partition supplied by MNIST
- [x] Image classifier (lightweight CNN)
- [x] Evaluation across individual classes
- [x] Confusion matrix chart
- [x] Sample inference with predicted labels and confidence
- [x] Training loss and accuracy progression
- [x] README with architecture and hyperparameters
- [x] Local execution using free open-source tools

## Notes
This is an educational image-classification project. The reported accuracy and prediction values are produced by actually running the training script on your computer; they are not pre-filled or guaranteed.
