"""
Spatial Frequency (SF) fusion metric.

SF measures the overall sharpness/detail of an image via row and
column gradient energy:

    RF = sqrt( mean( (I[i,j] - I[i,j-1])^2 ) )   row frequency
    CF = sqrt( mean( (I[i,j] - I[i-1,j])^2 ) )   column frequency
    SF = sqrt(RF^2 + CF^2)

Higher SF in the fused image (relative to the sharper of the two
sources at each region) indicates the fusion successfully concentrated
high-frequency detail rather than averaging it away.

Unlike MI, SF is evaluated on the fused image alone -- it does not
need access to the source images, which makes it useful as a no-reference
sharpness check (e.g. in the ablation study, to see whether removing
hyperedge aggregation produces a blurrier output).
"""
from __future__ import annotations

import numpy as np


def spatial_frequency(image: np.ndarray) -> float:
    """
    Compute spatial frequency for a single grayscale image.
    image: 2D array, any numeric dtype (cast to float64 internally).
    """
    img = image.astype(np.float64)

    row_diff = img[:, 1:] - img[:, :-1]
    col_diff = img[1:, :] - img[:-1, :]

    rf = np.sqrt(np.mean(row_diff ** 2)) if row_diff.size > 0 else 0.0
    cf = np.sqrt(np.mean(col_diff ** 2)) if col_diff.size > 0 else 0.0

    return float(np.sqrt(rf ** 2 + cf ** 2))
