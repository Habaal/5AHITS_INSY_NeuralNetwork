# Neuronales Netz von Grund auf (INSY, 5AHITS)

Umsetzung der Übung „Neural Network from Scratch“ (Teil 1–5).
Teil 6 (ESP32) ist laut Lehrer noch nicht dran.

## Ordner

| Ordner/Datei | Inhalt |
|---|---|
| `python/nn.py` | **Teil 1:** Klassen `Neuron`, `Layer`, `NeuralNetwork` – Training, Vorhersage, Export/Import |
| `python/part2_train.py` | **Teil 2:** XOR und 7-Segment-Decoder trainieren, exportieren, neu laden, testen |
| `python/part3_tensorflow.py` | **Teil 3:** dieselben Netze mit TensorFlow, Vergleich mit der eigenen Version |
| `c/nn.h`, `c/nn.c` | Netz in C (nur Forward/Predict), Laden aus `.txt` |
| `c/part4_txt.c` | **Teil 4:** Netz aus `output/7seg_net.txt` laden und testen |
| `c/part5_header.c` | **Teil 5:** Netz aus `output/7seg_net.h` initialisieren und testen |
| `web/` | Erklär-Website mit Animationen (`python web/server.py`) |
| `output/` | exportierte Netze (`.json`, `.txt`, `.h`) und Testausgaben (`partX_ergebnis.txt`) |

## Ausführen

Python 3 (Teil 1/2 brauchen keine Zusatzpakete, Teil 3 braucht `pip install tensorflow`):

```
cd python
python part2_train.py
python part3_tensorflow.py
```

C (gcc, oder ohne gcc: `pip install ziglang`, das Skript nimmt dann `python -m ziglang cc`):

```
c\build.bat
cd c
part4_txt.exe
part5_header.exe
```

## Erklär-Website

Interaktive Website mit Animationen zu allen Schritten (Neuron, Sigmoid, Forward, Gradientenabstieg,
Backpropagation, Live-Training, 7-Segment, Export, TensorFlow-Vergleich, C). Die Skripte lassen sich
dort auch starten – einzeln oder alle nacheinander.

```
python web/server.py
```

Öffnet automatisch http://127.0.0.1:8000 (nur Python-Standardbibliothek, nur lokal erreichbar).

## Wie das Netz funktioniert (kurz erklärt)

**Neuron:** rechnet `z = w1*x1 + w2*x2 + ... + bias` und gibt `a = sigmoid(z)` aus.
Sigmoid drückt jede Zahl auf einen Wert zwischen 0 und 1.

**Layer:** eine Liste von Neuronen. Alle bekommen dieselben Eingänge.

**Forward (Vorhersage):** Die Eingabe geht Schicht für Schicht durch, der Ausgang einer Schicht ist der
Eingang der nächsten. Der Ausgang der letzten Schicht ist die Vorhersage.

**Training (Backpropagation):**
1. Vorhersage berechnen.
2. Fehler pro Ausgang: `Vorhersage - Soll` (Ableitung der Kosten `1/2 * (Vorhersage - Soll)^2`).
3. Für jedes Neuron: `delta = Fehler * Ableitung von Sigmoid` (Sigmoid-Ableitung = `a * (1 - a)`).
4. Gewicht anpassen: `w = w - Lernrate * delta * Eingang`, Bias: `b = b - Lernrate * delta`.
5. Fehler für die Schicht davor: `Summe(delta * Gewicht)` – das ist die Kettenregel.
   Wichtig: mit den **alten** Gewichten rechnen, also vor dem Anpassen.
6. Das Ganze für jedes Beispiel, 20000 Epochen lang.

**Export:**
- `.json` – zum Wiederladen in Python (`NeuralNetwork.load`).
- `.txt` – lesbar, eine Zeile pro Neuron (`neuron 1 bias ... weights ...`); C liest sie mit `fscanf`.
- `.h` – alle Gewichte als C-Konstanten plus Funktion `SEG_NET_init()`, die das Netz damit aufbaut.
  So braucht das C-Programm (und später der ESP32) keine Datei zur Laufzeit.

**C-Version:** dieselben Structs (`Neuron`, `Layer`, `Network`), aber nur `net_forward`/`net_predict`,
kein Training. Die Gewichte kommen aus Python.

## Ergebnisse

Alle Werte stammen aus den Dateien in `output/`.

### Teil 2 (eigene Implementierung, 20000 Epochen, Lernrate 0.1)

| Netz | Architektur | MSE am Ende | richtige Bits | Export → Import gleich |
|---|---|---|---|---|
| XOR | 2-2-1 | 0.00127 | 4/4 | ja |
| 7-Segment | 4-16-7 | 0.00012 | 70/70 | ja |

Bei XOR bleibt der Fehler bis ca. Epoche 6000 bei 0.25 hängen und fällt dann bis Epoche 10000 stark ab –
das Netz muss erst „herausfinden“, dass es die zwei versteckten Neuronen unterschiedlich nutzen muss.

### Teil 3 (Vergleich mit TensorFlow)

Gleiche Architektur, 20000 Epochen, Lernrate 0.1, ein Beispiel pro Schritt (`batch_size=1`).
Volle Ausgabe: `output/part3_ergebnis.txt`.

| Netz | Version | MSE am Ende | richtige Bits | MSE < 0.01 ab Epoche | Trainingszeit |
|---|---|---|---|---|---|
| XOR 2-2-1 | eigenes Netz | 0.00127 | 4/4 | 10105 | 0.9 s |
| XOR 2-2-1 | TensorFlow | 0.168 | 3/4 | nie | 858 s |
| 7-Seg 4-16-7 | eigenes Netz | 0.00012 | 70/70 | 1270 | 20 s |
| 7-Seg 4-16-7 | TensorFlow | 0.00071 | 70/70 | 5633 | 1085 s |

**Was man daraus sieht:**
- **XOR mit TensorFlow ist hängen geblieben** (lokales Minimum): Für (0,0), (1,0) und (1,1) kommt
  jeweils 0.33 heraus. Mit nur 2 versteckten Neuronen hängt es stark von den zufälligen Startgewichten
  ab, ob XOR gelernt wird. Unser Netz hatte mit seinen Startwerten Glück, TensorFlow mit Seed 1 nicht.
  (Mit anderen Seeds wurde es nicht getestet.)
- **7-Segment** schaffen beide fehlerfrei. TensorFlow konvergiert langsamer, weil Keras den Fehler als
  `mean((y_hat - y)^2)` über die 7 Ausgänge rechnet. Der Gradient ist dadurch nur 2/7 so groß wie bei uns,
  das wirkt wie eine kleinere Lernrate.
- **Zeit:** TensorFlow ist hier sehr viel langsamer. Grund: Bei so winzigen Netzen und einem Beispiel pro
  Schritt kostet der Verwaltungsaufwand pro Schritt (Python → TensorFlow) mehr als die eigentliche
  Rechnung. Bei großen Netzen und großen Batches wäre TensorFlow klar schneller.

### Teil 4 und 5 (C)

Beide Programme liefern für die Ziffern 0–9 dieselben Werte wie Python (3 Nachkommastellen verglichen)
und 70/70 richtige Bits. Teil 4 liest `7seg_net.txt`, Teil 5 benutzt `7seg_net.h`.

## Werkzeuge

Python 3.10, TensorFlow (Teil 3), C99 kompiliert mit Zig `cc` (clang).
