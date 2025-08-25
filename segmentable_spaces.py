"""
This library enables one to visualize the 2D and 3D+ latent spaces of 1-layer
ReLU MLP layers. For latent spaces that are rank 2, use SegmentablePlane. For
latent spaces that are rank 3+, you will have to choose a 3D subspace in which
to visualize a cube of space, defined by 3 basis vectors.
"""
from enum import Enum
from typing import List, Optional, Tuple, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import scipy

BACKGROUND_COLOR = '#FCFBF8'
TOL = 1e-5

class UnsolvableSOEException(Exception):
    """Exception raised when a system of equations is unsolvable."""

    def __init__(self, message="The system of equations is unsolvable"):
        self.message = message
        super().__init__(self.message)


class PointType(Enum):
    INTERSECTION = 'intersection'
    VERTEX = 'vertex'


AnnotatedPoint: TypeAlias = tuple[np.ndarray, PointType]


class SegmentableSpace():
    """
    Contains common functions for SegmentablePlane and SegmentableCube.
    I would suggest looking at those classes to understand what they do.
    """
    @staticmethod
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
            b2 /= np.linalg.norm(b2)
            result = np.vstack([b1, b2])

            return result

        raise RuntimeError('Could not find a second basis vector')

    @staticmethod
    def get_intersection_coords(
        u: np.ndarray,
        v: np.ndarray,
        dim_idx: int
    ) -> Tuple[bool, None | np.ndarray]:
        """
        Given 2 points u and v, calculate point of intersection of uv with the
        coordinate hyperplane given by `dim_idx` == 0.

        NOTE:       Assume that u and v are NOT `dim_idx` == 0

        @return     (bool) has intersection, (np.ndarray) intersection point
        """
        assert u[dim_idx] != 0 and v[dim_idx] != 0, \
            'Do not call get_intersection_coords() on u, v where u, v could ' \
            f'be points of intersection: u: {u}, v: {v}'

        alpha = -u[dim_idx] / (v[dim_idx] - u[dim_idx])
        x = u + alpha * (v - u)

        # Does it intersect?
        x_to_u = u - x
        v_to_x = x - v
        # They must be pointing in same direction
        intersects = np.dot(x_to_u, v_to_x) > 0

        if intersects:
            return intersects, x
        else:
            return intersects, None

    @staticmethod
    def insert_intersection_for_edge(
        annotated_points: List[AnnotatedPoint],
        edge_idx_to_vertex_idxs: dict[int, List[int]],
        edge_idx: int,
        dim_idx: int,
    ) -> None:
        """
        Computes points of intersection between a line (an 'edge', since we're
        talking about planes and cubes) and the coordinate hyperplane given by
        `dim_idx` == 0, and:
        -   INSERTS them into `annotated_points`, and
        -   MUTATES `edge_idx_to_vertex_idxs`

        Example:
                    dim_idx
                        ^            o <-- vertex_idx: 4
                        |           /
                        |          / <---- edge_idx: 1
                        |         /
        ----------------+--------X------------> some other dim
                        |       /
                        |      /
                        |     o <-- vertex_idx: 5
                        |
        E.g. for the above edge (edge_idx: 1), we want to calculate X as it's
        the point of intersection between the edge and dim_idx == 0, and update
        
        edge_idx_to_vertex_idxs = { 1: [4, 5]} --> { 1: [4, P, 5]}
        annotated_points: (list of size P) --> (list of size P + 1)

        @param annotated_points:            A list of AnnotatedPoints
        @param edge_idx_to_vertex_idxs:     A mapping of which vertices belong
                                            to which edge.
        @param edge_idx:                    ID of the edge to calculate
                                            intersections for.
        @param dim_idx:                     Defines the coordinate hyperplane
                                            dim_idx == 0 to calculate
                                            intersections for.
        """
        vertex_idxs = edge_idx_to_vertex_idxs[edge_idx]
        assert len(vertex_idxs) == 2, \
            'Ad edge should only have 2 vertices if not yet inserted ' \
            'an intersection for'
        
        u = annotated_points[vertex_idxs[0]][0]
        v = annotated_points[vertex_idxs[1]][0]

        if np.allclose(u[dim_idx], 0):
            annotated_points[vertex_idxs[0]] = (u, PointType.INTERSECTION)
        if np.allclose(v[dim_idx], 0):
            annotated_points[vertex_idxs[1]] = (v, PointType.INTERSECTION)

        if annotated_points[vertex_idxs[0]][1] == PointType.INTERSECTION or \
            annotated_points[vertex_idxs[1]][1] == PointType.INTERSECTION:
            # Skip finding intersection because either edge is alr intersection
            intersects = False
        elif np.allclose(u[dim_idx], v[dim_idx]):
            # Parallel to dim_idx == 0
            intersects = False
        else:
            # Not parallel, could intersect. Solve SOE manually
            intersects, x = SegmentableSpace.get_intersection_coords(
                u, v, dim_idx
            )

        if intersects:
            # Add a new point to the list of annotated_points
            intersection_vertex_idx = len(annotated_points)
            annotated_points.append((x, PointType.INTERSECTION))
            # Update edge_idx_to_vertex_idxs
            updated_vertex_idxs = [
                vertex_idxs[0],
                intersection_vertex_idx,
                vertex_idxs[1]
            ]
            edge_idx_to_vertex_idxs[edge_idx] = updated_vertex_idxs


class SegmentablePlane(SegmentableSpace):
    r"""
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
              plane_vertices[2]
    
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

        self._break_into_segments_called = False
    
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
        basis_nd = SegmentableSpace.find_2_basis_vectors(self.W)
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
        r"""
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
        A_coordplane = np.zeros(shape=(1, self.plane_vertices.shape[-1]))
        A_coordplane[0, dim_idx] = 1
        b_coordplane = np.array([0])

        plane_center = np.mean(self.plane_vertices, axis=0)
        centered_plane_points = self.plane_vertices - plane_center[None,:]

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
    
    def plane_in_coordplane(self, dim_idx):
        """
        Checks if plane is contained within coordinate hyperplane given by
        dim_idx == 0
        """
        dim_idx_values = self.plane_vertices[:, dim_idx]
        return np.allclose(np.zeros_like(dim_idx_values), dim_idx_values)

    def insert_intersection_vertices(
            self,
            dim_idx: int
    ) -> List[AnnotatedPoint]:
        r"""
        Finds points of intersections between coordinate hyperplane defined
        by dim_idx == 0 and perimeter as described by self.plane_vertices.
        
        NOTE: In the case of an edge being ON that coordinate hyperplane, then
        the 2 vertices demarcating that line will just be marked as
        intersections.

        NOTE: We compute intersections here differently from in SegmentableCube
        because while we always visualize a cube in SegmentableCube, here, we
        visualize any arbitrary shape given by `self.plane_vertices`, which
        can be non-convex, or involve multiple consecutive vertices / edges
        lying on the coordinate hyperplane, which requires us to prune
        redundant points of intersection.

        Example: `R` is the redundant point here:

                dim_idx
                    ^ o
                    |/ \ 
                    /   \ 
                   /|    \ 
        ----o---R-o-+-----\------> some other dim
             \      |      \ 
              \     |       o
               \    |      /
                o---------o   
                    |
        
        To do so, it is better to keep a
        list of in-perimeter-order vertices, rather than to book-keep by
        mapping edge_idxs to vertex_idxs.

        @returns a list of AnnotatedPoints
        """
        annotated_points = []
        # Start with the first
        if np.allclose(self.plane_vertices[0, dim_idx], 0):
            annotated_points.append(
                (self.plane_vertices[0], PointType.INTERSECTION)
            )
        else:
            annotated_points.append(
                (self.plane_vertices[0], PointType.VERTEX)
            )

        for vertex_idx in range(1, self.num_vertices + 1):
            # Get edge
            u_i = vertex_idx - 1
            v_i = vertex_idx % self.num_vertices
            u = self.plane_vertices[u_i]
            v = self.plane_vertices[v_i]

            if np.allclose(v[dim_idx], 0):
                # v is an intersection, we append it to annotated_points,
                # unless v_i == 0, because that's already accounted for
                if v_i != 0:
                    annotated_points.append((v, PointType.INTERSECTION))
                # Doesn't make sense to compute point of intersection
                # between u and v. Next!
                continue
            elif np.allclose(u[dim_idx], 0):
                intersects = False
            elif np.allclose(u[dim_idx], v[dim_idx]):
                # Parallel to dim_idx == 0
                intersects = False
            else:
                intersects, x = SegmentableSpace.get_intersection_coords(
                    u, v, dim_idx
                )

            if intersects:
                annotated_points.append((x, PointType.INTERSECTION))
            
            if v_i != 0:
                annotated_points.append((v, PointType.VERTEX))
            
        # Now we have to prune consecutive intersections.
        # If there are 3 consecutive intersections, remove middle.
        running_consecutive_count = 0
        vertice_idxs_to_remove = []
        for i, (p, p_type) in enumerate(annotated_points):
            if p_type == PointType.INTERSECTION:
                running_consecutive_count += 1

                if running_consecutive_count >= 3:
                    vertice_idxs_to_remove.append(i - 1)
            else:
                running_consecutive_count = 0
        
        # Wrap around, because the above for loop doesn't detect
        # consecutive intersections that involve the last and first points
        if running_consecutive_count == 2 \
            and annotated_points[0][1] == PointType.INTERSECTION:
            vertice_idxs_to_remove.append(len(annotated_points) - 1)

        # Remove those redundant vertices
        while vertice_idxs_to_remove:
            next_idx = vertice_idxs_to_remove.pop()
            annotated_points.pop(next_idx)
        
        return annotated_points
    
    def separate_into_arcs(
            self,
            annotated_points: List[AnnotatedPoint],
    ) -> List[List[AnnotatedPoint]]:
        r"""
        Given annotated_points, a list that looks like: [
            ({coords}, PointType.VERTEX),
            ({coords}, PointType.INTERSECTION), 
            ...
        ]

        Want to return arcs book-ended by intersections:
        [
            [
                ({coords}, PointType.INTERSECTION),
                ...
                ({coords}, PointType.INTERSECTION)
            ],
            [
                ({coords}, PointType.INTERSECTION),
                ...
                ({coords}, PointType.INTERSECTION)
            ]
        ]

        Conceptually:
              o----o
             /      \             o----o          x        x
            x        x    ---->  /      \    +   /          \ 
           /          \         x        x      o------------o
          o------------o
        """
        arcs = []
        curr_arc: List[AnnotatedPoint] = []

        # Start from an intersection: find one.
        start_idx = 0
        for i, (p, p_type) in enumerate(annotated_points):
            if p_type == PointType.INTERSECTION:
                start_idx = i
                break
        
        curr_arc = [annotated_points[start_idx]]
        for i in range(start_idx + 1, start_idx + len(annotated_points)):
            p_idx = i % len(annotated_points)

            p, p_type = annotated_points[p_idx]
            if p_type == PointType.INTERSECTION:
                # Close off the current arc, and begin the next with this
                curr_arc.append((p, p_type))
                arcs.append(curr_arc)
                curr_arc = [(p, p_type)]
            else:
                curr_arc.append((p, p_type))

        # Wrap around
        curr_arc.append(annotated_points[start_idx])
        arcs.append(curr_arc)

        return arcs

    def filter_out_empty_arcs(
            self,
            arcs: List[List[AnnotatedPoint]]
    ) -> List[List[AnnotatedPoint]]:
        """
        Simply removes area 0 arcs. Remember that since we already pruned
        redundant intersections, if an arc has more than 2 points, it has a
        non-intersection => non-empty.
        """
        filtered_arcs = []
        for arc in arcs:
            if len(arc) == 2 \
                and arc[0][1] == PointType.INTERSECTION \
                and arc[1][1] == PointType.INTERSECTION:
                continue
            else:
                filtered_arcs.append(arc)

        return filtered_arcs

    def break_into_segments(
            self,
            dim_idx: int
    ) -> Tuple[List[List[AnnotatedPoint]], List[List[AnnotatedPoint]]]:
        r"""
        The (potentially non-convex) plane defined by `self.plane_vertices` can
        be segmented into multiple pieces via the coordinate hyperplane defined
        by `dim_idx` == 0:

          o       o
         / \     / \          o         o       x---x   x---x
        x---x---x---x   -->  / \   +   / \   +   \   \ /   /
         \   \ /   /        x---x     x---x       \   o   /
          \   o   /                                o-----o
           o-----o

                            <------------->     <----------->
                            "top segments"      "bottom segments"

        This is the function that is the entrypoint for most use-cases.

        @return:    tuple[top segments, bottom segments], where:
                    top segments is a list of segments, where each segment
                    is a list of AnnotatedPoints. Same for bottom segments
        """
        if self._break_into_segments_called:
            raise RuntimeError(
                'Do not call break_into_segments() multiple times on '
                'the same SegmentablePlane object'
            )
        self._break_into_segments_called = True

        # Exceedingly unlikely case: plane is entirely in the coordinate
        # hyperplane given by `dim_idx` == 0
        if self.plane_in_coordplane(dim_idx):
            return [], []
        
        dim_idx_values = self.plane_vertices[:, dim_idx]

        # Case: plane is on one side of coord hyperplane: no intersections
        if all(dim_idx_values >= 0):
            return [[(p, PointType.VERTEX) for p in self.plane_vertices]], []
        if all(dim_idx_values <= 0):
            return [], [[(p, PointType.VERTEX) for p in self.plane_vertices]]

        # Now we know there's some non-trivial intersection line. 

        # 1:    First, we find the equation of the line of intersection
        _, direction = self.get_plane_coordplane_intersection_line(dim_idx)

        # 2:    Then, we gather arcs
        annotated_points = self.insert_intersection_vertices(dim_idx)
        arcs = self.separate_into_arcs(annotated_points)
        arcs = self.filter_out_empty_arcs(arcs)

        # 3:    Then, we separate the arcs based on which side of the
        #       intersection.
        top_arcs = []
        bot_arcs = []
        for arc in arcs:
            # Find a non-intersection to see which side of the intersection
            # line it's on
            non_intersection = None
            for p, p_type in arc:
                if p_type == PointType.VERTEX:
                    non_intersection = p
                    break
            
            assert isinstance(non_intersection, np.ndarray), \
                'Did not find a non-intersection (VERTEX). ' \
                'filter_out_empty_arcs() first.'
            
            if non_intersection[dim_idx] > 0:
                top_arcs.append(arc)
            else:
                bot_arcs.append(arc)

        # 4:    We can't just assume that arcs will form a segment with the
        #       line of intersection like so:
        #               o---o       
        #              /     \ <-- arc
        #         - - x - - - x - - line of intersection
        #
        #       because there could also be holes due to the non-convexity:
        #         - - x - x - x - x - - - line of intersection
        #              \   \ /   /
        #               \   o   /
        #                o-----o 
        #
        #       So we have to look for arcs that are enclosed within other arcs
        #       that would create a hole.
        top_segments = []
        bot_segments = []
        for arc_group, segment_group in zip(
            [top_arcs, bot_arcs], 
            [top_segments, bot_segments]
        ):
            # Enumerate over all arcs, and try to find an enclosing arc
            # If found --> found piece with hole, otherwise found normal piece
            used_arc_idxs = set()
            for i, arc_a in enumerate(arc_group):
                if i in used_arc_idxs:
                    continue

                arc_a_start, arc_a_start_type = arc_a[0]
                arc_a_end, arc_a_end_type = arc_a[-1]

                # Sanity check that arc starts and ends with intersection
                assert arc_a_start_type == PointType.INTERSECTION, \
                    f'Arc[0] not intersection: {arc_a}'
                assert arc_a_end_type == PointType.INTERSECTION, \
                    f'Arc[-1] not intersection: {arc_a}'

                # Try to find enclosed arc
                found_hole = False
                for j, arc_b in enumerate(arc_group):
                    if i == j:
                        continue
                
                    arc_b_start, arc_b_start_type = arc_b[0]
                    arc_b_end, arc_b_end_type = arc_b[-1]

                    # Sanity check that arc starts and ends with intersection
                    assert arc_b_start_type == PointType.INTERSECTION, \
                        f'Arc[0] not intersection: {arc_b}'
                    assert arc_b_end_type == PointType.INTERSECTION, \
                        f'Arc[-1] not intersection: {arc_b}'

                    # Look to see if b is in a. For this to be true,
                    # a_start --> b_start must be the same direction as b_end --> a_end
                    starts_positive = np.dot(arc_b_start - arc_a_start, direction) > 0
                    ends_positive = np.dot(arc_a_end - arc_b_end, direction) > 0

                    if (starts_positive and ends_positive) or \
                        (not starts_positive and not ends_positive):
                        found_hole = True

                        segment = arc_a + arc_b[::-1]
                        segment_group.append(segment)
                        used_arc_idxs.add(i)
                        used_arc_idxs.add(j)
                        break

                if not found_hole:
                    segment_group.append(arc_a)
                    used_arc_idxs.add(i)

        return top_segments, bot_segments

    def convert_to_2d(
            self,
            segments: List[List[np.ndarray]] | List[List[Tuple[AnnotatedPoint]]] | List[np.ndarray],
            bias: np.ndarray,
            basis: np.ndarray
    ) -> List[np.ndarray]:
        """
        Since this class deals with a rank-2 subspace (plane) that lives in a
        higher, d-dimensional universe, we have thus far been dealing with
        d-dimensional coordinates. This function visualizes the plane in 2D
        with `bias` as the origin and `basis` as the axes.

        NOTE:   It may be tempting to assume that using Gram Schmidt on the
                plane_vertices (aka calling find_2_basis_vectors()) will always
                provide a good basis, but this is not true. For example, in
                auto-encoders:
                        y = ReLU(W_dec @ W_enc @ x + b)
                the features that you would like to visualize are the columns
                of W_enc, but applying W_dec effectively reinterprets W_enc
                in terms of the columns of W_dec, so W_dec is a natural
                choice of basis here. W_dec and the Gram-Schmidt-ed basis
                will span the same subspace, but W_dec is not normalized. And
                indeed, if you used the Gram-Schmidt-ed basis, you will have
                visualize a warped (linearly) version of W_enc.

        @param segments:    The segments. Accept 3 formats for each segment:
                            1)  List of unannotated points (1D np.ndarray)
                            2)  List of annotated points (AnnotatedPoint)
                            3)  A matrix of unannotated points (2D np.ndarray)
        @param bias:        (np.ndarray) the "origin" of the plane, shape (d,)
        @param basis:       (np.ndarray) of shape (2, d)

        @return:            The segments, but in the 2D world defined by `bias`
                            and `basis`. The format of each segment is also
                            a 2D matrix, i.e. (num_vertices, 2)
        """
        assert len(basis.shape) == 2 and basis.shape[0] == 2, 'Check basis'

        if len(segments) == 0:
            return []
        
        # Check type. We want to "un-annotate" it if it's annotated
        first_segment = segments[0]
        first_segment_p = first_segment[0]
        if isinstance(first_segment_p, Tuple):
            converted_segments = []
            for segment in segments:
                converted_segments.append([p[0] for p in segment])
            segments = converted_segments

        down_projector = basis.T @ np.linalg.inv(basis @ basis.T)
        down_projected_segments = []
        for segment in segments:
            # Stack all the segments into a np matrix for easy down-projection
            if isinstance(segment, List):
                vertices_np  = np.vstack(segment)
            elif isinstance(segment, np.ndarray):
                vertices_np = segment
            else:
                raise TypeError(f'Want list / np.ndarray, got {type(segment)}')
            
            vertices_np = vertices_np - bias[None,:]
            down_projected_vertices_np = vertices_np @ down_projector
            down_projected_segments.append(down_projected_vertices_np)

        return down_projected_segments
 

def segment_and_plot(
        plane_vertices: np.ndarray,
        W_enc: np.ndarray,
        W_dec: np.ndarray,
        bias: np.ndarray,
        Z: int = 3,
        plot_title: Optional[str] = None
    ):
    """
    Create and show a plot of a 2-dimensional latent space, whose shape
    is given by the in-order traversal of `plane_vertices`.

    @param plane_vertices:  (num_vertices, larger_dim) outline of 2D plane
    @param W_enc:           (2, larger_dim) encoder of auto-encoder
    @param W_dec:           (larger_dim, 2) decoder of auto-encoder
    @param bias:            (larger_dim,) bias of auto-encoder
    @param Z:               Plot radius
    @param plot_title:      Plot title
    """
    _, ax = plt.subplots()

    larger_dim = W_enc.shape[0]
    cmap = plt.get_cmap('coolwarm', larger_dim)
    for dim_idx in range(larger_dim):
        sp = SegmentablePlane(plane_vertices, W_enc, bias, reorient=True)
        top_segments, _ = sp.break_into_segments(dim_idx)
        top_segments_2d = sp.convert_to_2d(top_segments, bias=bias, basis=W_dec)

        # Plot the polygon
        for seg in top_segments_2d:
            polygon = plt.Polygon(seg, color=cmap(dim_idx), edgecolor=None, alpha=0.5)
            ax.add_patch(polygon)

    # # Set equal aspect ratio
    ax.set_facecolor(BACKGROUND_COLOR)
    Z = 3
    ax.set_aspect('equal')
    ax.set_xlim((-Z,Z))
    ax.set_ylim((-Z,Z))
    ax.set_title(plot_title)

    plt.show()