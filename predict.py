"""Run a prediction using the trained MNIST CNN."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

MODEL_PATH = os.path.join("outputs", "mnist_cnn.keras")

def main():
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from MNIST test data.")
    parser.add_argument("--index", type=int, default=0, help="Test image index from 0 to 9999")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run: python train.py")

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if not 0 <= args.index < len(x_test):
        raise ValueError(f"Index must be between 0 and {len(x_test)-1}")

    image = x_test[args.index]
    normalized = image.astype("float32") / 255.0
    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(normalized.reshape(1, 28, 28, 1), verbose=0)[0]
    predicted = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    print(f"Ground-truth label : {int(y_test[args.index])}")
    print(f"Predicted label    : {predicted}")
    print(f"Confidence         : {confidence*100:.2f}%")

    plt.imshow(image, cmap="gray")
    plt.title(
        f"Actual: {int(y_test[args.index])} | Predicted: {predicted}\n"
        f"Confidence: {confidence*100:.2f}%"
    )
    plt.axis("off")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    save_path = os.path.join("outputs", f"prediction_index_{args.index}.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Prediction image saved to {save_path}")

if __name__ == "__main__":
    main()
