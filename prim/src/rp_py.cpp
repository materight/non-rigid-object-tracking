#include <iostream>
#include <math.h>
#include "rp.h"
#include "py_helper.h"


extern "C" void rp(uchar *img, uint *imgShape, double *segmentMask,  uint nProposals, double *alpha, uint alphaSize, bool *out) {    

    // Load image
    const Image I(img, std::vector<uint> (imgShape, imgShape + 3), RGB);

    // Execute segmentation in different colorspaces
    Colorspace cspaces[4] = { LAB, Opponent, rg, HSV };
    int nProposalPerSpace = nProposals / 4;
    for (int i=0; i<4; i++) {
        // Load params
        const Params params = ParamsFromPy(nProposalPerSpace, alpha, alphaSize, cspaces[i]);

        // Execute random Prim.
        RP(I, segmentMask, params, out + (i * nProposalPerSpace * I.w() * I.h()));
    }
}