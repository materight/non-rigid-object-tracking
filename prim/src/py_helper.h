#ifndef PY_HELPER_H
#define PY_HELPER_H

#include <vector>
#include "params.h"

//TODO: read params from yaml file

Params ParamsFromPy(uint nProposals, double *alpha, uint alphaSize, Colorspace colorspace) {
    Params p;

    // Number of object proposals
    p.setNProposals(nProposals);

    // Image colorspace
    p.setColorspace(colorspace);

    // Superpixels segementation params
    Params::SpParams spParams;
    spParams.c_ = 100;
    spParams.min_size_ = 50;
    spParams.sigma_ = 0.5;
    p.setSpParams(spParams);

    // Alpha values, trained from VOC07
    std::vector<double> alphaVector(alpha, alpha + alphaSize);
    p.setAlpha(alphaVector);

    // Similarity weights
    Params::FWeights fWeights;
    fWeights.wBias_ = 3.0017;
    fWeights.wCommonBorder_ = -1.0029;
    fWeights.wLABColorHist_ = -2.6864;
    fWeights.wSizePer_ = -2.3655;
    p.setFWeights(fWeights);

    // Random seed
    p.setRSeedForRun(10690970);

    // Verbose output
    p.setVerbose(false);

    return p;
}

#endif