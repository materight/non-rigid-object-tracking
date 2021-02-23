#include <iostream>
#include <math.h>
#include "rp.h"
#include "py_helper.h"


extern "C" void rp(uchar *img, uint *imgShape, double *segmentMask,  uint nProposals, double *alpha, uint alphaSize, bool *out) {    

    // Load image
    const Image I(img, std::vector<uint> (imgShape, imgShape + 3), RGB);

    // Load params
    const Params params = ParamsFromPy(nProposals, alpha, alphaSize);

    // Execute random Prim.
    RP(I, segmentMask, params, out);
}