/* Teil 5: Netz aus dem Header 7seg_net.h initialisieren (keine Datei zur Laufzeit) und testen. */
#include <stdio.h>

#include "7seg_net.h"   /* von nn.py erzeugt, liegt in ../output */
#include "nn.h"

int main(void)
{
    Network net = {0};

    if (SEG_NET_init(&net) != 0) {
        fprintf(stderr, "Fehler: kein Speicher fuer das Netz\n");
        return 1;
    }
    printf("Initialisiert aus 7seg_net.h (%d Eingaenge, %d Schichten)\n\n", net.n_inputs, net.n_layers);

    int correct = test_7seg(&net);
    net_free(&net);
    return correct == 70 ? 0 : 1;
}
