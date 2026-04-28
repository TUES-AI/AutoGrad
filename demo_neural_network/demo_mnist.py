import gzip
import random
import struct
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt

from AutoGrad import Value as V
from NeuralNetwork import Neural_Network as NN
from Optimizer import Optimizer


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
DATA_DIR = Path(__file__).resolve().parent / "data" / "mnist"
PLOT_PATH = Path(__file__).resolve().parent / "mnist_demo.png"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

TRAIN_LIMIT = 300
TEST_LIMIT = 50
EPOCHS = 100
BATCH_SIZE = 10


def download_mnist():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"downloading {filename}")
            urllib.request.urlretrieve(MNIST_URL + filename, path)


def read_images(filename, limit=None):
    with gzip.open(DATA_DIR / filename, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image file: {filename}")

        count = min(count, limit or count)
        image_size = rows * cols
        images = []
        for _ in range(count):
            pixels = f.read(image_size)
            images.append([V(pixel / 255.0) for pixel in pixels])
        return images


def read_labels(filename, limit=None):
    with gzip.open(DATA_DIR / filename, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label file: {filename}")

        count = min(count, limit or count)
        labels = []
        for label in f.read(count):
            labels.append([V(1.0 if digit == label else 0.0) for digit in range(10)])
        return labels


def batches(xs, ys, batch_size):
    indexes = list(range(len(xs)))
    random.shuffle(indexes)
    for start in range(0, len(indexes), batch_size):
        batch_indexes = indexes[start : start + batch_size]
        yield [xs[i] for i in batch_indexes], [ys[i] for i in batch_indexes]


def predict_digit(model, image):
    outputs = model.forward([image])[0]
    return max(range(10), key=lambda i: outputs[i].value)


def label_digit(label):
    return max(range(10), key=lambda i: label[i].value)


def accuracy(model, images, labels):
    correct = 0
    for image, label in zip(images, labels):
        correct += predict_digit(model, image) == label_digit(label)
    return correct / len(images)


def image_values(image):
    return [pixel.value for pixel in image]


def plot_demo(losses, accuracies, model, images, labels):
    fig = plt.figure(figsize=(11, 7))
    grid = fig.add_gridspec(3, 4)

    ax_loss = fig.add_subplot(grid[0, :2])
    ax_loss.plot(range(1, len(losses) + 1), losses, marker="o")
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE")

    ax_acc = fig.add_subplot(grid[0, 2:])
    ax_acc.plot(range(1, len(accuracies) + 1), accuracies, marker="o", color="green")
    ax_acc.set_title("Test Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)

    for i, (image, label) in enumerate(zip(images[:8], labels[:8])):
        ax = fig.add_subplot(grid[1 + i // 4, i % 4])
        prediction = predict_digit(model, image)
        expected = label_digit(label)
        ax.imshow([image_values(image)[row * 28 : (row + 1) * 28] for row in range(28)], cmap="gray")
        ax.set_title(f"pred {prediction} / label {expected}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    print(f"saved demo plot to {PLOT_PATH}")
    plt.show()


download_mnist()

# This tiny AutoGrad engine is pure Python, so defaults are intentionally small.
train_images = read_images(FILES["train_images"], limit=TRAIN_LIMIT)
train_labels = read_labels(FILES["train_labels"], limit=TRAIN_LIMIT)
test_images = read_images(FILES["test_images"], limit=TEST_LIMIT)
test_labels = read_labels(FILES["test_labels"], limit=TEST_LIMIT)

model = NN([28 * 28,16,16, 10])
optimizer = Optimizer(model, lr=0.005)
loss_history = []
accuracy_history = []

for epoch in range(EPOCHS):
    total_loss = 0
    step_count = 0

    for xs, ys in batches(train_images, train_labels, batch_size=BATCH_SIZE):
        loss = optimizer.loss(xs, ys)
        total_loss += loss.value
        step_count += 1
        optimizer.update_params(loss)

    epoch_loss = total_loss / step_count
    epoch_accuracy = accuracy(model, test_images, test_labels)
    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_accuracy)

    print(
        f"epoch {epoch + 1} "
        f"loss {epoch_loss:.4f} "
        f"test_accuracy {epoch_accuracy:.2%}"
    )

for image, label in zip(test_images[:10], test_labels[:10]):
    print(f"predicted {predict_digit(model, image)} expected {label_digit(label)}")

plot_demo(loss_history, accuracy_history, model, test_images, test_labels)
