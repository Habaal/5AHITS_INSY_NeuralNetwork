#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int net_alloc(Network *net, int n_layers, const int *sizes, const int *activations)
{
    net->n_layers = n_layers;
    net->n_inputs = sizes[0];
    net->layers = calloc(n_layers, sizeof(Layer));
    if (!net->layers)
        return -1;
    for (int l = 0; l < n_layers; l++) {
        Layer *layer = &net->layers[l];
        layer->n_neurons = sizes[l + 1];
        if (layer->n_neurons < 1 || layer->n_neurons > NN_MAX_NEURONS)
            return -1;
        layer->activation = activations[l];
        layer->neurons = calloc(layer->n_neurons, sizeof(Neuron));
        if (!layer->neurons)
            return -1;
        for (int n = 0; n < layer->n_neurons; n++) {
            layer->neurons[n].n_inputs = sizes[l];   /* Eingänge = Neuronen der Schicht davor */
            layer->neurons[n].weights = calloc(sizes[l], sizeof(float));
            if (!layer->neurons[n].weights)
                return -1;
        }
    }
    return 0;
}

void net_free(Network *net)
{
    if (!net->layers)
        return;
    for (int l = 0; l < net->n_layers; l++) {
        if (!net->layers[l].neurons)
            continue;
        for (int n = 0; n < net->layers[l].n_neurons; n++)
            free(net->layers[l].neurons[n].weights);
        free(net->layers[l].neurons);
    }
    free(net->layers);
    net->layers = NULL;
}

static int parse_activation(const char *name)
{
    if (strcmp(name, "tanh") == 0)
        return ACT_TANH;
    if (strcmp(name, "relu") == 0)
        return ACT_RELU;
    return ACT_SIGMOID;
}

int net_load_txt(Network *net, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;

    int n_layers, n_inputs;
    if (fscanf(f, " layers %d inputs %d", &n_layers, &n_inputs) != 2 || n_layers < 1) {
        fclose(f);
        return -1;
    }

    /* Die Schichtgrößen stehen verteilt in der Datei, darum schichtweise anlegen. */
    net->n_layers = n_layers;
    net->n_inputs = n_inputs;
    net->layers = calloc(n_layers, sizeof(Layer));
    if (!net->layers) {
        fclose(f);
        return -1;
    }

    int prev = n_inputs;
    for (int l = 0; l < n_layers; l++) {
        Layer *layer = &net->layers[l];
        char act[16];
        if (fscanf(f, " layer %*d neurons %d activation %15s", &layer->n_neurons, act) != 2
            || layer->n_neurons < 1 || layer->n_neurons > NN_MAX_NEURONS)
            goto error;
        layer->activation = parse_activation(act);
        layer->neurons = calloc(layer->n_neurons, sizeof(Neuron));
        if (!layer->neurons)
            goto error;
        for (int n = 0; n < layer->n_neurons; n++) {
            Neuron *neuron = &layer->neurons[n];
            neuron->n_inputs = prev;
            neuron->weights = calloc(prev, sizeof(float));
            if (!neuron->weights)
                goto error;
            if (fscanf(f, " neuron %*d bias %f weights", &neuron->bias) != 1)
                goto error;
            for (int w = 0; w < prev; w++)
                if (fscanf(f, "%f", &neuron->weights[w]) != 1)
                    goto error;
        }
        prev = layer->n_neurons;
    }
    fclose(f);
    return 0;

error:
    fclose(f);
    net_free(net);
    return -1;
}

static float activate(int activation, float z)
{
    switch (activation) {
    case ACT_TANH:
        return tanhf(z);
    case ACT_RELU:
        return z > 0 ? z : 0;
    default:
        return 1.0f / (1.0f + expf(-z));
    }
}

void net_forward(Network *net, const float *inputs)
{
    const float *in = inputs;
    float buffer[NN_MAX_NEURONS];   /* Ausgänge der vorigen Schicht */

    for (int l = 0; l < net->n_layers; l++) {
        Layer *layer = &net->layers[l];
        for (int n = 0; n < layer->n_neurons; n++) {
            Neuron *neuron = &layer->neurons[n];
            float z = neuron->bias;
            for (int w = 0; w < neuron->n_inputs; w++)
                z += neuron->weights[w] * in[w];
            neuron->output = activate(layer->activation, z);
        }
        for (int n = 0; n < layer->n_neurons; n++)
            buffer[n] = layer->neurons[n].output;
        in = buffer;
    }
}

int net_predict(Network *net, const float *inputs, float *out)
{
    net_forward(net, inputs);
    Layer *last = &net->layers[net->n_layers - 1];
    for (int n = 0; n < last->n_neurons; n++)
        out[n] = last->neurons[n].output;
    return last->n_neurons;
}

int test_7seg(Network *net)
{
    static const int expected[10][7] = {
        {1, 1, 1, 1, 1, 1, 0}, {0, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 1, 1, 0, 1}, {1, 1, 1, 1, 0, 0, 1},
        {0, 1, 1, 0, 0, 1, 1}, {1, 0, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 1, 1, 1}, {1, 1, 1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 0, 1, 1},
    };
    int correct = 0;

    printf("Ziffer  Eingang    Ausgang (A-G)                              Bits     erwartet  \n");
    for (int d = 0; d < 10; d++) {
        float in[4] = {(float)((d >> 3) & 1), (float)((d >> 2) & 1), (float)((d >> 1) & 1), (float)(d & 1)};
        float out[7];
        int ok = 1;
        net_predict(net, in, out);

        printf("%d       [%d,%d,%d,%d]  ", d, (int)in[0], (int)in[1], (int)in[2], (int)in[3]);
        for (int s = 0; s < 7; s++)
            printf("%.3f ", out[s]);
        printf("  ");
        for (int s = 0; s < 7; s++) {
            int bit = out[s] >= 0.5f;
            printf("%d", bit);
            if (bit == expected[d][s])
                correct++;
            else
                ok = 0;
        }
        printf("  ");
        for (int s = 0; s < 7; s++)
            printf("%d", expected[d][s]);
        printf("   %s\n", ok ? "OK" : "FALSCH");
    }
    printf("Richtige Bits: %d/70\n", correct);
    return correct;
}
