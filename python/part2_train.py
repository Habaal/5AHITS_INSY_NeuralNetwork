"""Teil 2: XOR und 7-Segment-Decoder trainieren, exportieren, wieder laden und testen."""

import os
import random

from nn import NeuralNetwork

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
TOLERANCE = 1e-9  # geladenes Netz muss (fast) exakt gleich rechnen

XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_Y = [[0], [1], [1], [0]]

# Ziffer 0-9 als 4 Bit -> Segmente A-G
SEG_X = [[(d >> 3) & 1, (d >> 2) & 1, (d >> 1) & 1, d & 1] for d in range(10)]
SEG_Y = [
    [1, 1, 1, 1, 1, 1, 0],  # 0
    [0, 1, 1, 0, 0, 0, 0],  # 1
    [1, 1, 0, 1, 1, 0, 1],  # 2
    [1, 1, 1, 1, 0, 0, 1],  # 3
    [0, 1, 1, 0, 0, 1, 1],  # 4
    [1, 0, 1, 1, 0, 1, 1],  # 5
    [1, 0, 1, 1, 1, 1, 1],  # 6
    [1, 1, 1, 0, 0, 0, 0],  # 7
    [1, 1, 1, 1, 1, 1, 1],  # 8
    [1, 1, 1, 1, 0, 1, 1],  # 9
]


def check(name, trained, loaded, inputs, targets):
    """Vergleicht trainiertes und geladenes Netz und zählt richtige Bits (Schwelle 0.5)."""
    print(f"\n--- Test {name}: trainiert vs. geladen ---")
    correct_bits = total_bits = 0
    same = True
    for x, y in zip(inputs, targets):
        p_trained = trained.predict(x)
        p_loaded = loaded.predict(x)
        same &= all(abs(a - b) <= TOLERANCE for a, b in zip(p_trained, p_loaded))
        bits = [1 if p >= 0.5 else 0 for p in p_loaded]
        correct_bits += sum(b == t for b, t in zip(bits, y))
        total_bits += len(y)
        print(f"{x} -> {[round(p, 3) for p in p_loaded]}  Bits {bits}  erwartet {y}"
              f"  {'OK' if bits == y else 'FALSCH'}")
    print(f"Export/Import identisch: {same}")
    print(f"Richtige Bits: {correct_bits}/{total_bits}")


def main():
    os.makedirs(OUT, exist_ok=True)
    random.seed(1)

    # Schritt 1 + 2: XOR (2-2-1)
    print("=== XOR trainieren (2-2-1, 20000 Epochen, alpha 0.1) ===")
    xor = NeuralNetwork([2, 2, 1])
    xor.train(XOR_X, XOR_Y, epochs=20000, learning_rate=0.1)
    xor.dump_json(os.path.join(OUT, "xor_net.json"))
    xor.dump_txt(os.path.join(OUT, "xor_net.txt"))
    xor.dump_initfile(os.path.join(OUT, "xor_net.h"), name="XOR_NET")
    check("XOR", xor, NeuralNetwork.load(os.path.join(OUT, "xor_net.json")), XOR_X, XOR_Y)

    # Schritt 3 + 4: 7-Segment (4-16-7)
    print("\n=== 7-Segment trainieren (4-16-7, 20000 Epochen, alpha 0.1) ===")
    seg = NeuralNetwork([4, 16, 7])
    seg.train(SEG_X, SEG_Y, epochs=20000, learning_rate=0.1)
    seg.dump_json(os.path.join(OUT, "7seg_net.json"))
    seg.dump_txt(os.path.join(OUT, "7seg_net.txt"))
    seg.dump_initfile(os.path.join(OUT, "7seg_net.h"), name="SEG_NET")
    check("7-Segment", seg, NeuralNetwork.load(os.path.join(OUT, "7seg_net.json")), SEG_X, SEG_Y)


if __name__ == "__main__":
    main()
