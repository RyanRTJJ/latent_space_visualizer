import matplotlib.pyplot as plt
import numpy as np

from segmentable_spaces import segment_and_plot
from segmentable_spaces import SegmentablePlane

BACKGROUND_COLOR = '#FCFBF8'

def get_unit_2d_vectors_regularly_spaced(num_vecs: int) -> np.ndarray:
    phases = np.linspace(0, 2 * np.pi, num_vecs + 1)[:-1]
    phases_2d = np.vstack([np.cos(phases), np.sin(phases)])
    W = phases_2d.T
    return W

def get_rigged_bias(num_dims: int) -> np.ndarray:
    """
    Creates a uniformly negative bias that is negative by *JUST* the right
    amount. What's the "right amount"? Read this section of my blogpost:
    https://amagibaba.com/posts/viewing-latent-spaces/#the-role-of-b
    """
    b = np.ones(shape=(num_dims,)) * (-np.cos(2 * np.pi / num_dims))
    print(b)
    return b

if __name__ == '__main__':
    num_feats = 6

    # Shape (6, 2) and (2, 6) respectively
    W_enc = get_unit_2d_vectors_regularly_spaced(num_vecs=num_feats)
    W_dec = W_enc.T

    # Shape (6,)
    b = get_rigged_bias(num_dims=num_feats)

    # Assume each column of W_enc corresponds cleanly to a feature.
    # To get feature 1, we do e1 @ W_enc 
    # To get all features, we do identity @ W_enc @ W_dec
    plane_vertices = np.eye(num_feats) @ W_enc @ W_dec + b

    # Create a figure and axis
    segment_and_plot(
        plane_vertices,
        W_enc,
        W_dec,
        bias=b,
        Z=3,
        plot_title='Example plot'
    )
