"""Neuronales Netz von Grund auf (Teil 1).

Aufbau: NeuralNetwork -> Layer -> Neuron.
Training mit Backpropagation und stochastischem Gradientenabstieg (ein Beispiel pro Schritt),
Kostenfunktion: J = 1/2 * (y_hat - y)^2  ->  dJ/dy_hat = y_hat - y
"""

import json
import math
import random


# ---------------------------------------------------------------------------
# Aktivierungsfunktionen: (Funktion, Ableitung ausgedrückt über den Ausgang a)
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


ACTIVATIONS = {
    "sigmoid": (_sigmoid, lambda a: a * (1.0 - a)),
    "tanh": (math.tanh, lambda a: 1.0 - a * a),
    "relu": (lambda z: max(0.0, z), lambda a: 1.0 if a > 0 else 0.0),
}


class Neuron:
    def __init__(self, n_inputs, activation="sigmoid", weights=None, bias=None):
        self.weights = weights if weights is not None else [random.uniform(-1, 1) for _ in range(n_inputs)]
        self.bias = bias if bias is not None else random.uniform(-1, 1)
        self.activation = activation
        self.last_inputs = []
        self.output = 0.0

    def activate(self, inputs):
        """output = f(dot(inputs, weights) + bias); merkt sich Ein- und Ausgang für backward."""
        f, _ = ACTIVATIONS[self.activation]
        z = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
        self.last_inputs = list(inputs)
        self.output = f(z)
        return self.output

    def update(self, delta, learning_rate):
        """Gradient für w_i ist delta * x_i, für den Bias ist er delta."""
        for i, x in enumerate(self.last_inputs):
            self.weights[i] -= learning_rate * delta * x
        self.bias -= learning_rate * delta


class Layer:
    def __init__(self, n_neurons, n_inputs, activation="sigmoid"):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]

    def forward(self, inputs):
        return [n.activate(inputs) for n in self.neurons]

    def backward(self, errors, learning_rate):
        """errors[j] = dJ/da_j für Neuron j dieser Schicht.

        Gibt dJ/dx_i für die vorige Schicht zurück (Kettenregel):
        dJ/dx_i = Summe_j delta_j * w_ji
        """
        n_inputs = len(self.neurons[0].weights)
        prev_errors = [0.0] * n_inputs
        for neuron, error in zip(self.neurons, errors):
            _, df = ACTIVATIONS[neuron.activation]
            delta = error * df(neuron.output)            # dJ/dz
            for i in range(n_inputs):                    # vor dem Update rechnen (alte Gewichte!)
                prev_errors[i] += delta * neuron.weights[i]
            neuron.update(delta, learning_rate)
        return prev_errors


class NeuralNetwork:
    def __init__(self, layer_sizes, activation="sigmoid"):
        """layer_sizes z. B. [2, 2, 1]: 2 Eingänge, 2 versteckte Neuronen, 1 Ausgang."""
        self.layer_sizes = list(layer_sizes)
        self.layers = [
            Layer(layer_sizes[i + 1], layer_sizes[i], activation)
            for i in range(len(layer_sizes) - 1)
        ]

    # --- Forward -----------------------------------------------------------
    def forward(self, inputs):
        a = inputs
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def predict(self, inputs):
        return self.forward(inputs)

    # --- Training ----------------------------------------------------------
    def train(self, inputs, targets, epochs=20000, learning_rate=0.1, verbose=True):
        """Trainiert mit SGD; gibt die Liste der mittleren Fehler (MSE) pro Epoche zurück."""
        history = []
        for epoch in range(epochs):
            total = 0.0
            for x, y in zip(inputs, targets):
                y_hat = self.forward(x)
                errors = [p - t for p, t in zip(y_hat, y)]   # dJ/dy_hat
                total += sum(e * e for e in errors) / len(errors)
                for layer in reversed(self.layers):
                    errors = layer.backward(errors, learning_rate)
            mse = total / len(inputs)
            history.append(mse)
            if verbose and (epoch % (epochs // 10 or 1) == 0 or epoch == epochs - 1):
                print(f"Epoche {epoch:6d}  MSE {mse:.6f}")
        return history

    # --- Export / Import JSON ----------------------------------------------
    def to_dict(self):
        return {
            "layer_sizes": self.layer_sizes,
            "layers": [
                [{"activation": n.activation, "bias": n.bias, "weights": n.weights} for n in layer.neurons]
                for layer in self.layers
            ],
        }

    def dump_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        net = cls(data["layer_sizes"])
        for layer, layer_data in zip(net.layers, data["layers"]):
            layer.neurons = [
                Neuron(len(n["weights"]), n["activation"], list(n["weights"]), n["bias"])
                for n in layer_data
            ]
        return net

    # --- Export TXT (menschenlesbar, auch von C mit fscanf lesbar) ---------
    def dump_txt(self, path):
        """Format:
        layers <Anzahl Schichten ohne Eingang>
        inputs <Anzahl Eingänge>
        layer <nr> neurons <anzahl> activation <name>
        neuron <nr> bias <b> weights <w1> <w2> ...
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"layers {len(self.layers)}\n")
            f.write(f"inputs {self.layer_sizes[0]}\n")
            for li, layer in enumerate(self.layers, start=1):
                f.write(f"\nlayer {li} neurons {len(layer.neurons)} activation {layer.neurons[0].activation}\n")
                for ni, n in enumerate(layer.neurons, start=1):
                    w = " ".join(f"{x:.10f}" for x in n.weights)
                    f.write(f"neuron {ni} bias {n.bias:.10f} weights {w}\n")

    # --- Export C-Header ----------------------------------------------------
    def dump_initfile(self, path, name="NET"):
        """Schreibt eine .h-Datei mit allen Konstanten des Netzes und einer Init-Funktion.

        Pro Schicht L: <name>_L<L>_BIAS[neuronen] und <name>_L<L>_WEIGHTS[neuronen][eingaenge].
        <name>_init(Network *net) baut daraus ein Netz (Structs aus c/nn.h).
        """
        guard = f"{name.upper()}_H"
        lines = [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "/* automatisch erzeugt von nn.py */",
            "",
            '#include "nn.h"',
            "",
            f"#define {name}_NUM_LAYERS {len(self.layers)}",
            f"#define {name}_NUM_INPUTS {self.layer_sizes[0]}",
            f"#define {name}_NUM_OUTPUTS {self.layer_sizes[-1]}",
            "",
            f"static const int {name}_LAYER_SIZES[{len(self.layer_sizes)}] = "
            f"{{{', '.join(map(str, self.layer_sizes))}}};",
            "",
        ]
        for li, layer in enumerate(self.layers, start=1):
            n_out, n_in = len(layer.neurons), len(layer.neurons[0].weights)
            biases = ", ".join(f"{n.bias:.10f}f" for n in layer.neurons)
            lines.append(f"static const float {name}_L{li}_BIAS[{n_out}] = {{{biases}}};")
            lines.append(f"static const float {name}_L{li}_WEIGHTS[{n_out}][{n_in}] = {{")
            for n in layer.neurons:
                lines.append("    {" + ", ".join(f"{w:.10f}f" for w in n.weights) + "},")
            lines.append("};")
            lines.append("")

        # Init-Funktion: Speicher anlegen und alle Gewichte/Biases aus den Konstanten kopieren
        acts = ", ".join(f"ACT_{layer.neurons[0].activation.upper()}" for layer in self.layers)
        lines += [
            f"static const int {name}_ACTIVATIONS[{len(self.layers)}] = {{{acts}}};",
            "",
            f"/* Gibt 0 bei Erfolg zurück, -1 wenn kein Speicher frei ist. */",
            f"static int {name}_init(Network *net)",
            "{",
            f"    if (net_alloc(net, {name}_NUM_LAYERS, {name}_LAYER_SIZES, {name}_ACTIVATIONS) != 0)",
            "        return -1;",
        ]
        for li, layer in enumerate(self.layers, start=1):
            lines += [
                f"    for (int n = 0; n < {len(layer.neurons)}; n++) {{",
                f"        net->layers[{li - 1}].neurons[n].bias = {name}_L{li}_BIAS[n];",
                f"        for (int w = 0; w < {len(layer.neurons[0].weights)}; w++)",
                f"            net->layers[{li - 1}].neurons[n].weights[w] = {name}_L{li}_WEIGHTS[n][w];",
                "    }",
            ]
        lines += ["    return 0;", "}", "", f"#endif /* {guard} */"]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
