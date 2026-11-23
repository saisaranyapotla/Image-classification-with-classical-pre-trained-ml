"""Train a CNN on the MNIST handwritten-digit benchmark dataset."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = [str(i) for i in range(10)]

def main():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values from [0, 255] to [0, 1] and add channel dimension.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Training images: {x_train.shape}; Test images: {x_test.shape}")
    print("Building lightweight CNN...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    print("Training model...")
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    model.save(os.path.join(OUTPUT_DIR, "mnist_cnn.keras"))

    # Export training curves.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="Training loss")
    axes[0].plot(history.history["val_loss"], label="Validation loss")
    axes[0].set_title("Loss progression")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Training accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Accuracy progression")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)

    # Confusion matrix for all ten classes.
    cm = confusion_matrix(y_test, predictions, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("MNIST Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Save per-class precision, recall and F1-score.
    report = classification_report(
        y_test, predictions, labels=list(range(10)),
        target_names=CLASS_NAMES, digits=4
    )
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Prediction log and image demo: show test samples with true/predicted labels + confidence.
    sample_count = 12
    sample_indices = np.arange(sample_count)
    sample_probs = probabilities[sample_indices]
    sample_preds = predictions[sample_indices]
    sample_conf = np.max(sample_probs, axis=1)

    with open(os.path.join(OUTPUT_DIR, "prediction_logs.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_index", "ground_truth", "predicted_label", "confidence"])
        for idx, true_label, pred_label, conf in zip(sample_indices, y_test[:sample_count], sample_preds, sample_conf):
            writer.writerow([int(idx), int(true_label), int(pred_label), f"{float(conf):.4f}"])

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i].squeeze(), cmap="gray")
        ax.set_title(
            f"True: {y_test[i]} | Pred: {sample_preds[i]}\nConfidence: {sample_conf[i]*100:.1f}%",
            fontsize=9
        )
        ax.axis("off")
    fig.suptitle("Sample MNIST Predictions", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test loss: {test_loss:.4f}\nTest accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test images: {len(y_test)}\nEpochs: {EPOCHS}\nBatch size: {BATCH_SIZE}\n")

    print("\nTraining complete.")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"All charts, model and logs saved in: {OUTPUT_DIR}/")
    print("\nClassification report:\n", report)

if __name__ == "__main__":
    main()
