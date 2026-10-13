/* Teil 4: Netz aus der .txt-Datei laden und den 7-Segment-Decoder testen. */
#include <stdio.h>

#include "nn.h"

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "../output/7seg_net.txt";
    Network net = {0};

    if (net_load_txt(&net, path) != 0) {
        fprintf(stderr, "Fehler: %s konnte nicht geladen werden\n", path);
        return 1;
    }
    printf("Geladen: %s (%d Eingaenge, %d Schichten)\n\n", path, net.n_inputs, net.n_layers);

    int correct = test_7seg(&net);
    net_free(&net);
    return correct == 70 ? 0 : 1;
}
