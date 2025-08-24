from typing import List, Optional, Tuple
import unittest

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from segmentable_spaces import segment_and_plot
from segmentable_spaces import PointType
from segmentable_spaces import SegmentablePlane
from segmentable_spaces import UnsolvableSOEException

class SegmentablePlaneTests(unittest.TestCase):
    """Unit Tests for SegmentablePlane"""
    def test_raise_if_not_plane(self):
        # Test 1: Rank 3, not a plane
        plane_points = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]
        ])
        with self.assertRaises(AssertionError):
            _ = SegmentablePlane(plane_points, W=plane_points)

    def test_accept_plane(self):
        # Rank 2: a plane
        plane_points = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)

    def test_plane_in_coordplane(self):
        # Test the condition where plane is entirely contained in dim_idx == 0
        plane_points = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        self.assertTrue(sp.plane_in_coordplane(dim_idx=3))
        self.assertFalse(sp.plane_in_coordplane(dim_idx=2))

    def test_unsolvable_SOE(self):
        plane_points = np.array([
            [1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        with self.assertRaises(UnsolvableSOEException):
            sp.get_plane_coordplane_intersection_line(2)

    def test_correct_intersections_convex(self):
        """
                            ^
                            o
                        ....|....
                    ........|........
                o...........|...........o
                ............|............
                ............|............
        --------X-----------+-----------X------------>
                ............|............
                ............|............
                o...........|...........o
                    ........|........
                        ....|....
                            o
        """
        plane_points = np.array([
            [0, 0, 1, 0, 0],
            [0.866, 0, 0.5, 0, 0],
            [0.866, 0, -0.5, 0, 0],
            [0, 0, -1, 0, 0],
            [-0.866, 0, -0.5, 0, 0],
            [-0.866, 0, 0.5, 0, 0],
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        points_and_intersections = sp.insert_intersection_vertices(dim_idx=2)
        self.assertEqual(len(points_and_intersections), 8)

        expected = [(p, PointType.VERTEX) for p in plane_points]
        expected.extend([
            (np.array([0.866, 0, 0, 0, 0]), PointType.INTERSECTION),
            (np.array([-0.866, 0, 0, 0, 0]), PointType.INTERSECTION),
        ])
        for expected_point, expected_type in expected:
            found_point = False
            for p, p_type in points_and_intersections:
                if np.allclose(p, expected_point) and p_type == expected_type:
                    found_point = True
                    break
            self.assertTrue(found_point)

    def test_correct_intersections_double_up(self):
        r"""
        hexagon, but 2 plane points should be turned into intersections
                            ^
                            |
                    o.......|.....o
                  ..........|........
                 ...........|.........
               .............|...........
        ------X-------------+-----------X------>
               .............|...........
                 ...........|.........
                  ..........|........
                    o.......|.....o
                            |
        """
        plane_points = np.array([
            [0.5, 0, 0.866, 0, 0],
            [1, 0, 0, 0, 0],
            [0.5, 0, -0.866, 0, 0],
            [-0.5, 0, -0.866, 0, 0],
            [-1, 0, 0, 0, 0],
            [-0.5, 0, 0.866, 0, 0],
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        points_and_intersections = sp.insert_intersection_vertices(dim_idx=2)
        self.assertEqual(len(points_and_intersections), 6)

        expected = [(p, PointType.VERTEX) for p in plane_points]
        expected[1] = (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION)
        expected[4] = (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION)
        for expected_point, expected_type in expected:
            found_point = False
            for p, p_type in points_and_intersections:
                if np.allclose(p, expected_point) and p_type == expected_type:
                    found_point = True
                    break
            self.assertTrue(found_point)

    def test_correct_intersections_pacman(self):
        """
        (non-convex) pacman facing up 
                            ^
                            |
                     o      |      o 
        ------------x-x-----+-----x-x---------->
                  ......    |   .......
                ........... | ..........
               o............o...........X
                ............|...........
                  ..........|.........
                   .........|........
                     o......|.....o
                            |
        """
        plane_points = np.array([
            [0.5, 0, -0.5 + 0.866, 0, 0],
            [1, 0, -0.5, 0, 0],
            [0.5, 0, -0.5 - 0.866, 0, 0],
            [-0.5, 0, -0.5 - 0.866, 0, 0],
            [-1, 0, -0.5, 0, 0],
            [-0.5, 0, -0.5 + 0.866, 0, 0],
            [0, 0, -0.5, 0, 0],
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        points_and_intersections = sp.insert_intersection_vertices(dim_idx=2)
        self.assertEqual(len(points_and_intersections), 11)

        X_OUTER_INT_POINT = 0.71131639722
        X_INNER_INT_POINT = 0.28868360277
        expected = [(p, PointType.VERTEX) for p in plane_points]
        expected.insert(1, (np.array([X_OUTER_INT_POINT, 0, 0, 0, 0]), PointType.INTERSECTION))
        expected.insert(6, (np.array([-X_OUTER_INT_POINT, 0, 0, 0, 0]), PointType.INTERSECTION))
        expected.insert(8, (np.array([-X_INNER_INT_POINT, 0, 0, 0, 0]), PointType.INTERSECTION))
        expected.insert(10, (np.array([X_INNER_INT_POINT, 0, 0, 0, 0]), PointType.INTERSECTION))
        for expected_point, expected_type in expected:
            found_point = False
            for p, p_type in points_and_intersections:
                if np.allclose(p, expected_point) and p_type == expected_type:
                    found_point = True
                    break
            self.assertTrue(found_point)

    def test_remove_triplet_colinear(self):
        """
        3 intersections in a line --> drop the middle one
                    ^
                    |
                    |
        ----x-------x-------x-------->
              ......|......
                ....|....
                  ..|..
                    o
                    |
        """
        plane_points = np.array([
            [-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        points_and_intersections = sp.insert_intersection_vertices(dim_idx=2)
        self.assertEqual(len(points_and_intersections), 3)
        expected = [
            (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
            (np.array([0, 0, -1, 0, 0]), PointType.VERTEX),
        ]
        for expected_point, expected_type in expected:
            found_point = False
            for p, p_type in points_and_intersections:
                if np.allclose(p, expected_point) and p_type == expected_type:
                    found_point = True
                    break
            self.assertTrue(found_point)

    def check_arcs_equivalence(
            self,
            arcs: List[List[Tuple[np.ndarray, str]]],
            expected_arcs: List[List[Tuple[np.ndarray, str]]]
        ):
        """Helper function to test arcs == expected_arcs"""
        for arc in arcs:
            found_matching_arc = False
            for e_arc in expected_arcs:
                if len(arc) == len(e_arc):
                    # Compare all the points in this arc
                    found_point_mismatch = False
                    for (p, p_type), (e_p, e_p_type) in zip(arc, e_arc):
                        if p_type != e_p_type or not np.allclose(p, e_p):
                            found_point_mismatch = True
                            break

                    if found_point_mismatch:
                        # Move on to next arc to try and find match
                        continue
                    else:
                        found_matching_arc = True

                    if found_matching_arc:
                        break
                else:
                    # Arcs have different number of points, could not possible match
                    pass
            self.assertTrue(found_matching_arc, 'Unexpected arc:\n' + '\n'.join([str(p) for p in arc]))

    def test_separate_into_arcs_1(self):
        r"""
        simple 2 arc problem where some plane points are intersections,
        starting from non-intersection
                    ^
                    |
                    o  <-- start here         o
                   /|\                       / \ 
                  / | \                     /   \ 
                 /  |  \                   /     \ 
        --------x---+---x-------->  -->   x       x  +  x       x
                 \  |  /                                 \     /
                  \ | /                                   \   /
                   \|/                                     \ /
                    o                                       o
                    |
        """
        plane_points = np.array([
            [-1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0]
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        points_and_intersections = sp.insert_intersection_vertices(dim_idx=2)
        arcs = sp.separate_into_arcs(points_and_intersections)
        self.assertEqual(len(arcs), 2)

        expected_arcs = [
            [
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, -1, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
        ]
        self.check_arcs_equivalence(arcs, expected_arcs)

    def test_separate_into_arcs_2(self):
        """
        simple 2 arc problem but dont start from intersection
                    ^
                    |
                    o                         o
                   /|\                       / \ 
                  / | \  start              /   \ 
                 /  |  \ here              /     \ 
        --------x---+---x-------->  -->   x       x  +  x       x
                 \  |  /                                 \     /
                  \ | /                                   \   /
                   \|/                                     \ /
                    o                                       o
                    |

        """
        plane_points = np.array([
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [-1, 0, 0, 0, 0],
        ])

        sp = SegmentablePlane(plane_points, W=plane_points)
        points_and_intersections = sp.insert_intersection_vertices(dim_idx=2)
        arcs = sp.separate_into_arcs(points_and_intersections)
        self.assertEqual(len(arcs), 2)

        expected_arcs = [
            [
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, -1, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
        ]
        self.check_arcs_equivalence(arcs, expected_arcs)

    def test_separate_into_arcs_3(self):
        """
        Cat ears: testing 3 consecutive intersections
        
              o           o               o         o
             ...         ...             / \       / \ 
        ----x---x---x---x---x---> --->  x   x  +  x   x  + ...cont'd below
              .............
                .........
                  .....
                    o

        
        ...cont'd  +    x               x  +  x------x  (the middle piece)
                          \           /
                            \       /
                              \   /
                                o
        
        """
        plane_points = np.array([
            [-3, 0, 0, 0, 0],
            [-2, 0, 1, 0, 0],
            [-1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 0, 1, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, -2, 0, 0]
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        points_and_intersections = sp.insert_intersection_vertices(dim_idx=2)
        arcs = sp.separate_into_arcs(points_and_intersections)
        self.assertEqual(len(arcs), 4)
        expected_arcs = [
            [
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([-2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, -2, 0, 0]), PointType.VERTEX),
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
        ]
        self.check_arcs_equivalence(arcs, expected_arcs)

    def test_separate_into_arcs_4(self):
        """
        Cat ears but points are permuted (rotated by 1) such that we
        don't start from intersection
        """
        plane_points = np.array([
            [-2, 0, 1, 0, 0],
            [-1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 0, 1, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, -2, 0, 0],
            [-3, 0, 0, 0, 0],
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        points_and_intersections = sp.insert_intersection_vertices(dim_idx=2)
        arcs = sp.separate_into_arcs(points_and_intersections)
        self.assertEqual(len(arcs), 4)
        expected_arcs = [
            [
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, -2, 0, 0]), PointType.VERTEX),
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([-2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
        ]
        self.check_arcs_equivalence(arcs, expected_arcs)

    def test_separate_into_arcs_5(self):
        r"""
        bean shape with hole
             o       o               o         o 
            / \     / \             / \       / \
        ---x---x---x---x--->  -->  x   x  +  x   x  +  x           x  +  x   x
            \   \ /   /                                 \         /       \ /
             \   o   /                                   \       /         o
              o-----o                                     o-----o
        """
        plane_points = np.array([
            [-3, 0, 0, 0, 0],
            [-2, 0, 1, 0, 0],
            [-1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 0, 1, 0, 0],
            [3, 0, 0, 0, 0],
            [1, 0, -2, 0, 0],
            [-1, 0, -2, 0, 0],
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        points_and_intersections = sp.insert_intersection_vertices(dim_idx=2)
        arcs = sp.separate_into_arcs(points_and_intersections)
        self.assertEqual(len(arcs), 4)

        expected_arcs = [
            [
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([-2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, -1, 0, 0]), PointType.VERTEX),
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([1, 0, -2, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, -2, 0, 0]), PointType.VERTEX),
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
        ]
        self.check_arcs_equivalence(arcs, expected_arcs)

    def test_filter_out_empty_arcs(self):
        """
        Tests that empty arcs get thrown away. The shape is the same as in
        test_separate_into_arcs_3, and we want to throw out the 'middle piece'
        """
        MOCK_plane_points = np.array([
            [-3, 0, 0, 0, 0],
            [-2, 0, 1, 0, 0],
            [-1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 0, 1, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, -2, 0, 0]
        ])
        sp = SegmentablePlane(MOCK_plane_points, W=MOCK_plane_points)
        input_arcs = [
            [
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, -2, 0, 0]), PointType.VERTEX),
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([-2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
        ]
        remaining_arcs = sp.filter_out_empty_arcs(input_arcs)
        expected_arcs = [
            [
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, -2, 0, 0]), PointType.VERTEX),
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([-2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
        ]
        self.check_arcs_equivalence(remaining_arcs, expected_arcs)

    def test_break_into_segments_1(self):
        """
        simple 2 segment problem
                    ^
                    |
                    o                           o
                  ..|..                       .....
                ....|....                   .........
              ......|......               .............
        ----x-------+-------x--->  -->  o...............o  +  o...............o
              ......|......                                     .............
                ....|....                                         .........
                  ..|..                                             .....
                    o                                                 o
                    |

        """
        plane_points = np.array([
            [-1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0]
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        top_segments, bot_segments = sp.break_into_segments(dim_idx=2)

        expected_top_segments = [
            [
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ]
        ]
        expected_bot_segments = [
            [
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, -1, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
        ]
        self.check_arcs_equivalence(top_segments, expected_top_segments)
        self.check_arcs_equivalence(bot_segments, expected_bot_segments)

    def test_break_into_segments_3(self):
        """
        Cat ears: testing 3 consecutive intersections
        
        
              o           o               o         o
             ...         ...             ...       ... 
        ----x---x---x---x---x---> --->  o...o  +  o...o  +  o...............o
              .............                                   .............
                .........                                       .........
                  .....                                           .....
                    o                                               o
        """
        plane_points = np.array([
            [-3, 0, 0, 0, 0],
            [-2, 0, 1, 0, 0],
            [-1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 0, 1, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, -2, 0, 0]
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        top_segments, bot_segments = sp.break_into_segments(dim_idx=2)

        expected_top_segments = [
            [
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([-2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
        ]
        expected_bot_segments = [
            [
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, -2, 0, 0]), PointType.VERTEX),
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
        ]
        self.check_arcs_equivalence(top_segments, expected_top_segments)
        self.check_arcs_equivalence(bot_segments, expected_bot_segments)

    def test_break_into_segments_5(self):
        """
        bean shape with hole
             o       o               o         o 
            ...     ...             ...       ...
        ---x---x---x---x--->  -->  o...o  +  o...o  +  o...o   o...o
            ..... .....                                 ..... .....
             ....o....                                   ....o....
              o.....o                                     o.....o
        """
        plane_points = np.array([
            [-3, 0, 0, 0, 0],
            [-2, 0, 1, 0, 0],
            [-1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 0, 1, 0, 0],
            [3, 0, 0, 0, 0],
            [1, 0, -2, 0, 0],
            [-1, 0, -2, 0, 0],
        ])
        sp = SegmentablePlane(plane_points, W=plane_points)
        top_segments, bot_segments = sp.break_into_segments(dim_idx=2)

        expected_top_segments = [
            [
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([-2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
            [
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([2, 0, 1, 0, 0]), PointType.VERTEX),
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
            ],
        ]
        expected_bot_segments = [
            [
                (np.array([3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([1, 0, -2, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, -2, 0, 0]), PointType.VERTEX),
                (np.array([-3, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([1, 0, 0, 0, 0]), PointType.INTERSECTION),
                (np.array([0, 0, -1, 0, 0]), PointType.VERTEX),
                (np.array([-1, 0, 0, 0, 0]), PointType.INTERSECTION),
            ][::-1],
        ]
        self.check_arcs_equivalence(top_segments, expected_top_segments)
        self.check_arcs_equivalence(bot_segments, expected_bot_segments)

def test_real_easy():
    """
    Creates a wheel in 7d.
    Objective: to observe a REGULAR aperture-looking thing with 7
    activation zones.
    """
    DIMS = 7
    phases = np.linspace(0, 2 * np.pi, DIMS + 1)[:-1]
    basis_2d = np.vstack([np.cos(phases), np.sin(phases)]) # Shape (2, 6)
    W_enc = basis_2d.T
    W_dec = basis_2d

    # Simulate W and b
    bias = np.ones(shape=(DIMS,)) * (-np.cos(2 * np.pi / DIMS))
    plane_vertices = W_enc @ W_dec + bias

    segment_and_plot(
        plane_vertices,
        W_enc,
        W_dec,
        bias,
        plot_title='easy'
    )

def test_real_medium():
    """
    Real example, non-convex.
    Objective: no spiky areas / lines; well-formed segments only.
    """
    W = np.array([
        [0.2101169228553772, 0.7183502912521362],
        [-0.5308800935745239, 0.20420998334884644],
        [0.3803485333919525, 0.2677828073501587],
        [-0.07176265120506287, -0.768387496471405],
        [-0.6902794241905212, -0.3854953646659851],
        [0.3035421669483185, 0.5389608144760132],
        [-0.7994059324264526, 0.06336105614900589]
    ])
    bias = np.array(
        [-0.00048912, -0.00048739, 0.00048982, 0.00048884, -0.0004895, -0.00048711, -0.00048939]
    )
    W_enc = W
    W_dec = W.T
    plane_vertices = W_enc @ W_dec + bias
    segment_and_plot(
        plane_vertices,
        W_enc,
        W_dec,
        bias,
        plot_title='medium'
    )

def test_real_hard_1():
    """
    Real example, non-convex.
    Objective: no spiky areas / lines; well-formed segments only.
    """
    W = np.array([
        [-0.1245,  1.0019],
        [ 0.7419, -0.7416],
        [-0.4328, -1.0311],
        [ 0.6840,  0.2993],
        [ 0.7809,  0.2965],
        [-0.0796,  0.0249],
        [-1.0300,  0.1045]
    ])
    W_enc = W
    W_dec = W.T
    bias = np.array(
        [-0.11733911, -0.27031204, -0.33977968, -0.15580186, -0.20498118, 0.14917181, -0.17040823]
    )
    plane_vertices = W_enc @ W_dec + bias
    segment_and_plot(
        plane_vertices,
        W_enc,
        W_dec,
        bias,
        plot_title='hard_1'
    )

def test_real_hard_2():
    """
    scaling issue
    """
    W = np.array([
        [ 0.18835023, -0.15468554],
        [ 0.2384949,   0.22723384],
        [ 0.4894902,  -0.94338584],
        [-0.17052586, -0.21439601],
        [-0.31407323, -0.16145296],
        [-0.2262537,  0.26467642],
        [ 0.35585225, -0.5705964 ]
    ])
    W_enc = W
    W_dec = W.T
    bias = np.array(
        [ 0.08013767, 0.08500358, -0.07837567, 0.07780223, 0.07753915, 0.08347972, -0.06036647]
    )
    plane_vertices = W_enc @ W_dec + bias
    segment_and_plot(
        plane_vertices,
        W_enc,
        W_dec,
        bias,
        plot_title='hard_2'
    )

def test_real_hard_3a():
    """flickering test case, seemingly non continuous jump"""
    W = np.array([
        [-1.0676248,   0.42004994],
        [ 1.0984012,   0.30490673],
        [-0.72475994, -0.88506037],
        [ 0.06078263, 1.0945939 ],
        [-0.00651505, -0.00622136],
        [ 0.626741,   -0.95563674]
    ])
    W_enc = W
    W_dec = W.T
    bias = np.array([-0.3611859, -0.34938422, -0.34529158, -0.31051394, 0.16661547, -0.34505215])
    plane_vertices = W_enc @ W_dec + bias
    segment_and_plot(
        plane_vertices,
        W_enc,
        W_dec,
        bias,
        plot_title='hard_3a'
    )

def test_real_hard_3b():
    """
    flickering test case, seemingly non continuous jump
    UPDATE: Non-issue here. Pentagon looks like it should be completely filled in but
    should be a kink (wedge hole) that corresponds to the vector very near 0, and that
    vector can switch positions very easily.
    """
    W = np.array([
        [-1.0966101e+00,  4.1120175e-01],
        [ 1.1237212e+00,  2.9777062e-01],
        [-7.3761868e-01, -9.1579598e-01],
        [ 5.5086672e-02,  1.1334900e+00],
        [ 6.5870507e-04,  4.2745648e-03],
        [ 6.3595384e-01, -9.8505217e-01]
    ])
    W_enc = W
    W_dec = W.T
    bias = np.array(
        [-0.39749995, -0.3805158, -0.4054155, -0.3460592, 0.16669956, -0.39936775]
    )
    plane_vertices = W_enc @ W_dec + bias
    segment_and_plot(
        plane_vertices,
        W_enc,
        W_dec,
        bias,
        plot_title='hard_3b'
    )

def test_real_hard_4a():
    """
    flickering test case, seemingly non continuous jump from line (4a) to plane (4b)
    RCA: return type of break_into_segments was wrong for case entirely above / below
    """
    W = np.array([
        [-1.6290927,  -0.4650574 ],
        [ 1.4694626,  -0.13723059],
        [-0.72139645, -1.52889   ],
        [-0.40359733,  0.99871486],
        [ 0.02470034,  0.13486336],
        [ 0.6754821,  -1.5483538 ]
    ])
    W_enc = W
    W_dec = W.T
    bias = np.array(
        [-1.8762457, -1.1855615, -1.8730507, -0.17881934, 0.22629191, -1.864316]
    )
    plane_vertices = W_enc @ W_dec + bias
    segment_and_plot(
        plane_vertices,
        W_enc,
        W_dec,
        bias,
        plot_title='hard_4a'
    )

def test_real_hard_4b():
    """
    flickering test case, seemingly non continuous jump from line (4a) to plane (4b)
    """
    W = np.array([
        [-1.6302938,  -0.49531165],
        [ 1.4872534,  -0.15338992],
        [-0.7076331,  -1.5457711 ],
        [-0.40557864,  0.9869476 ],
        [ 0.02274,     0.13838594],
        [ 0.68891484, -1.5526551 ]
    ])
    W_enc = W
    W_dec = W.T
    bias = np.array(
        [-1.909293, -1.2427725, -1.9056998, -0.15786676, 0.22903632, -1.8964984]
    )
    plane_vertices = W_enc @ W_dec + bias
    segment_and_plot(
        plane_vertices,
        W_enc,
        W_dec,
        bias,
        plot_title='hard_4b'
    )

def test_real_hard_5():
    """
    Single line; actually correct. No bug.
    """
    W = np.array([
        [-8.4958684e-01,  5.1396465e-01],
        [ 3.8483489e-01, -9.1539711e-01],
        [ 2.2630991e-01,  9.6639234e-01],
        [-7.5156498e-01, -6.4911705e-01],
        [ 1.8746563e-04,  4.9882807e-04],
        [ 9.8907191e-01,  8.3078325e-02],
        [ 8.5264641e-05,  6.6329394e-06]
    ])
    W_enc = W
    W_dec = W.T
    bias = np.array(
        [-0.18344198, -0.18354511, -0.18279961, -0.18369757, 0.13802636, -0.18288557, 0.13802624]
    )
    plane_vertices = W_enc @ W_dec + bias
    segment_and_plot(
        plane_vertices,
        W_enc,
        W_dec,
        bias,
        plot_title='hard_5'
    )

if __name__ == '__main__':
    test_real_easy()
    test_real_medium()
    test_real_hard_1()
    test_real_hard_2()
    test_real_hard_3a()
    test_real_hard_3b()
    test_real_hard_4a()
    test_real_hard_4b()
    test_real_hard_5()
    unittest.main()