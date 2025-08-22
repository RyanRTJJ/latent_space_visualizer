"""
This library enables one to visualize the 2D and 3D+ latent spaces of 1-layer
ReLU MLP layers. For latent spaces that are rank 2, use SegmentablePlane. For
latent spaces that are rank 3+, you will have to choose a 3D subspace in which
to visualize a cube of space, defined by 3 basis vectors.
"""
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import unittest

TOL = 1e-5

class UnsolvableSOEException(Exception):
    """Exception raised when a system of equations is unsolvable."""

    def __init__(self, message="The system of equations is unsolvable"):
        self.message = message
        super().__init__(self.message)

def find_2_basis_vectors(
        features: np.ndarray
) -> np.ndarray:
    """
    Uses Gram Schmidt to return 2 basis vectors that span the space in which
    `features` live based on the first 2 non-zero rows of `features`.

    @param features:    a (>= 2, dim) matrix of effective dimensionality >= 2

    @return:            a (2, dim) matrix
    """
    features_copy = features.copy()
    b1 = features_copy[0]
    b1 /= np.linalg.norm(b1)

    # Because this function is usually called on W, during the process of
    # learning, W can have multiple zero features, so we can't just blindly
    # take the first 2 rows
    for i in range(1, features_copy.shape[0]):
        if np.linalg.norm(features_copy[i]) == 0:
            continue

        b2 = features_copy[i] - (np.dot(features_copy[i], b1) * b1)
        b2 = np.linalg.norm(b2)
        result = np.vstack([b1, b2])

        return result

    raise RuntimeError('Could not find a second basis vector')

class SegmentablePlane():
    """
    Suppose you have a set of points living in n-dimensional space. Going from
    point to point allows us to trace a shape, and further suppose that this
    set of points is rank-2, which means that the shape is a 2D polygon, which
    we'll just call 'plane'. Further suppose that this plane is offset from
    the origin by a 'bias', such that it does not intersect with the origin,
    but may intersect with coordinate hyperplanes, e.g. in 3D:

                  dim 1
                    ^
                    |
        ------------+--------------------------> dim 0
                    |\         [5]
                    | \        x 
                    |  \   ..........
                [4] |  .\..............x plane_vertices[0]
                   x|....\.............
                  ++|.....\...........
                 +++|......B.........
                ++++|..............x
           [3] x++++|...........   plane_vertices[1]
                   +|.......
                    |  x
              plane_verrtices[2]
    
    Observe that the dim 1 == 0 coordinate plane intersects with the plane.
    
    Then, this class helps to compute the coordinates of intersection of
    the plane with the coordinate hyperplane of choice (specified by dim_idx)
    and return the various segments (those in the positive, colored by '.'s
    above, and those in the negative region, color by '+'s above).

    Because those intersection points / segment vertices live in n-d space,
    they are not useful for visualization. We hence also convert them back
    to 2D. To do so, we need to specify an origin, as well as 2 n-d basis
    vectors. In the context of an autoencoder:
        Code convention: y = ReLU(X @ W @ W.T + b), where W has shape (5, 2),
    if we define the origin as simply the bias (`b`), which is drawn in the
    diagram above, and set the basis as the rows of W.T, then we naturally
    are able to visualize the plane above (hexagon) in 2 dimensions, in a
    way that is faithful to the features (rows) of W. 
    """
    def __init__(
            self,
            plane_vertices: np.ndarray,
            W: np.ndarray,
            bias: Optional[np.ndarray]=None,
            reorient: bool=False,
    ):
        """
        @param plane_vertices:  (np.ndarray) a (num_vertices, dim) matrix
        @param W:               (np.ndarray) the W in y = ReLU(X @ W @ W.T + b)
        @param bias:            (np.ndarray) the b in y = ReLU(X @ W @ W.T + b)
        @param reorient:        (bool) True if you want to re-order the rows of
                                W (and plane_points too, correspondingly) in 
                                clockwise fashion starting from W[0]
        """
        W = W.astype(np.float64)
        if not isinstance(bias, np.ndarray):
            bias = np.zeros(shape=(W.shape[1],))
        bias = bias.astype(np.float64)
        plane_vertices = plane_vertices.astype(np.float64)

        self.W = W
        self.bias = bias
        self.plane_vertices = plane_vertices
        self.num_vertices = self.plane_vertices.shape[0]
    
        # Sanity check
        self.assert_2d()

        if reorient:
            self.reorient_W()
    
    def assert_2d(self):
        """Checks that the plane_vertices form a 2D plane"""
        plane_center = np.mean(self.plane_vertices, axis=0, keepdims=True)
        centered_plane_points = self.plane_vertices - plane_center

        plane_rank = np.linalg.matrix_rank(centered_plane_points, tol=TOL)
        assert plane_rank == 2, f'plane_vertices has rank {plane_rank}'

    def reorient_W(self):
        """Reorient W to be specified in clockwise fashion"""
        # Take first feature to be north start
        v = self.W[0].copy()
        v = v / np.linalg.norm(v)

        # Calculate angles
        cosines = self.W @ v
        cosines /= np.linalg.norm(self.W, axis=-1)
        cosines = np.clip(cosines, a_min=-1.0, a_max=1.0)
        # Have to clip because of numeric instability
        angles_rad = np.arccos(cosines)

        # Determine CW-ness by using sign of cross product with perp
        basis_nd = find_2_basis_vectors(self.W)
        perp = basis_nd[1]
        cross_prods = np.dot(self.W, perp)
        # Shift angles_rad to be between 0 and 2 * pi
        angles_rad = np.where(
            cross_prods < 0,
            2 * np.pi - angles_rad,
            angles_rad
        )

        # Sort
        ordering = np.argsort(angles_rad)
        self.W = self.W[ordering]
        self.plane_vertices = self.plane_vertices[ordering]
        self.reorient_ordering = ordering
    
    def get_plane_coordplane_intersection_line(
            self,
            dim_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets a point and direction (parametric form) that represents the line
        of intersection between the plane as outlined by taking self.W as the
        vertices, as the object described by dim_idx == 0:

        Visualization of the plane represented by W. Note that because the
        plane is defined by tracing self.plane_vertices, it is not
        guaranteed to be convex:

                  dim 1
                    ^
                    |
        ------------+--------------------------> dim 0
                    |\ bias
                    | \        6 
                    |  \   ..........
                    |  .\..............0
                   5|....\.............
                    | ....\...........
                    |    4...........
                    |..............1
               3....|...........
                   .|.......
                    |  2

        Approach: the coordinate hyperplane described by dim_idx == 0
        can also be described by this SOE:
        [0, ..., 0, 1, 0, ..., 0] @ [x_0, ..., x_idx, ... x_(n-1)] == [0]
                    ^                                                 ^
                position idx                                call this 'b'
        <---- call this 'A' ---->

        NOTE: The caller is expected to take care of the edge case where
        an edge of the plane lies ON the coordinate hyperplane provided.
        """
        # Construct SOE for coordinate hyperplane
        A_coordplane = np.zeros(shape=(1, self.W.shape[-1]))
        A_coordplane[0, dim_idx] = 1
        b_coordplane = np.array([0])

        plane_center = np.mean(self.plane_vertices, axis=0, keepdims=True)
        centered_plane_points = self.plane_vertices - plane_center

        # Construct SOE for plane
        # This SOE is just the cartesian form of equation of plane
        # Perform SVD on on the centered points
        _, s, vh = np.linalg.svd(centered_plane_points, full_matrices=False)

        # The last (n-2) elements of s MUST be 0 since plane is 2D
        last_few_s = s[2:]
        assert np.allclose(np.zeros_like(last_few_s), last_few_s, atol=1e-4), \
            f'too many > 0 singular values: {s}'
        
        # The last (n-2) right singular vectors are normal to the plane
        normals = vh[2:]
        b_plane = np.dot(normals, plane_center)

        # Combine the 2 SOEs and solve for intersection (null space)
        A = np.vstack((A_coordplane, normals))
        b = np.hstack((b_coordplane, b_plane))

        # Sanity check: does the system have an exact solution?
        rank_A = np.linalg.matrix_rank(A)
        rank_Ab = np.linalg.matrix_rank(np.column_stack((A, b)))

        if rank_A != rank_Ab:
            raise UnsolvableSOEException(f'This is unsolvable. A:\n{A},b:\n: {b}')
        
        # Find the null space of combined system
        null_space = scipy.linalg.null_space(A)

        if null_space.shape[1] != 1:
            # This intersection is not a line
            return None, None

        # Solve the system to find a point on the line
        point = scipy.linalg.lstsq(A, b)[0]
        direction = null_space[:, 0]

        return point, direction
    
