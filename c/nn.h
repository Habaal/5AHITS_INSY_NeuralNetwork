/* Neuronales Netz in C - nur Forward/Predict, kein Training (Teil 4 und 5). */
#ifndef NN_H
#define NN_H

#define NN_MAX_NEURONS 64   /* max. Neuronen pro Schicht (Puffer in net_forward) */

enum { ACT_SIGMOID, ACT_TANH, ACT_RELU };

typedef struct {
    int n_inputs;
    float *weights;   /* n_inputs Gewichte */
    float bias;
    float output;     /* letzter berechneter Ausgang */
} Neuron;

typedef struct {
    int n_neurons;
    int activation;   /* ACT_SIGMOID, ACT_TANH oder ACT_RELU */
    Neuron *neurons;
} Layer;

typedef struct {
    int n_layers;     /* Schichten ohne Eingangsschicht */
    int n_inputs;
    Layer *layers;
} Network;

/* sizes hat n_layers + 1 Einträge: Eingänge, dann Neuronen je Schicht. 0 = ok, -1 = Fehler. */
int net_alloc(Network *net, int n_layers, const int *sizes, const int *activations);
void net_free(Network *net);

/* Liest das .txt-Format aus nn.py (dump_txt). 0 = ok, -1 = Fehler. */
int net_load_txt(Network *net, const char *path);

/* Rechnet die Eingabe durch alle Schichten. */
void net_forward(Network *net, const float *inputs);

/* forward + Ausgänge der letzten Schicht nach out kopieren; gibt Anzahl der Ausgänge zurück. */
int net_predict(Network *net, const float *inputs, float *out);

/* Testet die Ziffern 0-9 gegen die 7-Segment-Tabelle; gibt Anzahl richtiger Bits (von 70) zurück. */
int test_7seg(Network *net);

#endif
