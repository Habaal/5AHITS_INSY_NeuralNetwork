"""Teil 3: XOR und 7-Segment mit TensorFlow trainieren und mit der eigenen Implementierung vergleichen.

Gleiche Architektur, gleiche Epochen (20000), gleiche Lernrate (0.1), SGD mit einem Beispiel pro Schritt.
Hinweis: Keras-MSE ist mean((y_hat - y)^2), unsere Kosten sind 1/2 * Summe (y_hat - y)^2.
Die Gradienten unterscheiden sich daher um den Faktor 2/Ausgänge (XOR: doppelt so groß, 7-Seg: 2/7).
"""

import os
import random
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TensorFlow-Infomeldungen ausblenden

import numpy as np
import tensorflow as tf

from nn import NeuralNetwork
from part2_train import SEG_X, SEG_Y, XOR_X, XOR_Y

EPOCHS = 20000
LEARNING_RATE = 0.1


def build_tf(sizes):
    model = tf.keras.Sequential([tf.keras.Input(shape=(sizes[0],))])
    for n in sizes[1:]:
        model.add(tf.keras.layers.Dense(n, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE), loss="mse")
    return model


def evaluate(predictions, targets):
    """Gibt (MSE, richtige Bits, alle Bits) zurück; Schwelle 0.5."""
    mse = correct = total = 0
    for p, y in zip(predictions, targets):
        mse += sum((a - b) ** 2 for a, b in zip(p, y)) / len(y)
        correct += sum((1 if a >= 0.5 else 0) == b for a, b in zip(p, y))
        total += len(y)
    return mse / len(targets), correct, total


def compare(name, sizes, inputs, targets):
    print(f"\n=== {name} {sizes}, {EPOCHS} Epochen, alpha {LEARNING_RATE} ===")

    random.seed(1)
    own = NeuralNetwork(sizes)
    t0 = time.perf_counter()
    own_hist = own.train(inputs, targets, epochs=EPOCHS, learning_rate=LEARNING_RATE, verbose=False)
    own_time = time.perf_counter() - t0
    own_pred = [own.predict(x) for x in inputs]

    tf.keras.utils.set_random_seed(1)
    model = build_tf(sizes)
    x, y = np.array(inputs, dtype="float32"), np.array(targets, dtype="float32")
    t0 = time.perf_counter()
    tf_hist = model.fit(x, y, epochs=EPOCHS, batch_size=1, shuffle=False, verbose=0).history["loss"]
    tf_time = time.perf_counter() - t0
    tf_pred = model.predict(x, verbose=0).tolist()

    print(f"{'Eingang':<14}{'erwartet':<24}{'eigenes Netz':<44}TensorFlow")
    for xi, yi, po, pt in zip(inputs, targets, own_pred, tf_pred):
        print(f"{str(xi):<14}{str(yi):<24}{str([round(v, 2) for v in po]):<44}{[round(v, 2) for v in pt]}")

    for label, pred, hist, secs in (("eigenes Netz", own_pred, own_hist, own_time),
                                    ("TensorFlow", tf_pred, tf_hist, tf_time)):
        mse, correct, total = evaluate(pred, targets)
        # Konvergenz: erste Epoche, ab der der Fehler unter 0.01 liegt
        conv = next((i for i, v in enumerate(hist) if v < 0.01), None)
        print(f"{label:<13} MSE {mse:.6f}  Bits {correct}/{total}  Zeit {secs:6.1f} s  "
              f"MSE<0.01 ab Epoche {conv if conv is not None else 'nie'}")


def main():
    compare("XOR", [2, 2, 1], XOR_X, XOR_Y)
    compare("7-Segment", [4, 16, 7], SEG_X, SEG_Y)


if __name__ == "__main__":
    main()
