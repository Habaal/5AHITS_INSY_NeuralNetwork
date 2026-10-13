#ifndef XOR_NET_H
#define XOR_NET_H

/* automatisch erzeugt von nn.py */

#include "nn.h"

#define XOR_NET_NUM_LAYERS 2
#define XOR_NET_NUM_INPUTS 2
#define XOR_NET_NUM_OUTPUTS 1

static const int XOR_NET_LAYER_SIZES[3] = {2, 2, 1};

static const float XOR_NET_L1_BIAS[2] = {2.6149192763f, -3.0630692515f};
static const float XOR_NET_L1_WEIGHTS[2][2] = {
    {-5.2188716945f, 5.4229929923f},
    {-5.6721284669f, 5.5630666137f},
};

static const float XOR_NET_L2_BIAS[1] = {3.6935137025f};
static const float XOR_NET_L2_WEIGHTS[1][2] = {
    {-7.8864043949f, 8.1839284416f},
};

static const int XOR_NET_ACTIVATIONS[2] = {ACT_SIGMOID, ACT_SIGMOID};

/* Gibt 0 bei Erfolg zurück, -1 wenn kein Speicher frei ist. */
static int XOR_NET_init(Network *net)
{
    if (net_alloc(net, XOR_NET_NUM_LAYERS, XOR_NET_LAYER_SIZES, XOR_NET_ACTIVATIONS) != 0)
        return -1;
    for (int n = 0; n < 2; n++) {
        net->layers[0].neurons[n].bias = XOR_NET_L1_BIAS[n];
        for (int w = 0; w < 2; w++)
            net->layers[0].neurons[n].weights[w] = XOR_NET_L1_WEIGHTS[n][w];
    }
    for (int n = 0; n < 1; n++) {
        net->layers[1].neurons[n].bias = XOR_NET_L2_BIAS[n];
        for (int w = 0; w < 2; w++)
            net->layers[1].neurons[n].weights[w] = XOR_NET_L2_WEIGHTS[n][w];
    }
    return 0;
}

#endif /* XOR_NET_H */
