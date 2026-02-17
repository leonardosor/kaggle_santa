"""
Python wrapper for Go geometry library using ctypes.

This module provides high-performance geometric operations by calling
compiled Go functions through a C-compatible interface.
"""

import ctypes
import os
import numpy as np
from typing import List, Tuple

# Find the shared library
_lib_path = os.path.join(os.path.dirname(__file__), '../golib/cmd/libgeometry/libgeometry.so')
if not os.path.exists(_lib_path):
    raise ImportError(f"Go library not found at {_lib_path}. Please build it first using 'make build-go'")

# Load the shared library
_lib = ctypes.CDLL(_lib_path)

# Define C function signatures

# PointInPolygonC(pointX, pointY float64, vertices *float64, numVertices int) int
_lib.PointInPolygonC.argtypes = [
    ctypes.c_double,  # pointX
    ctypes.c_double,  # pointY
    ctypes.POINTER(ctypes.c_double),  # vertices
    ctypes.c_int,  # numVertices
]
_lib.PointInPolygonC.restype = ctypes.c_int

# SegmentsIntersectC(s1StartX, s1StartY, s1EndX, s1EndY, s2StartX, s2StartY, s2EndX, s2EndY float64) int
_lib.SegmentsIntersectC.argtypes = [
    ctypes.c_double, ctypes.c_double,  # s1Start
    ctypes.c_double, ctypes.c_double,  # s1End
    ctypes.c_double, ctypes.c_double,  # s2Start
    ctypes.c_double, ctypes.c_double,  # s2End
]
_lib.SegmentsIntersectC.restype = ctypes.c_int

# PolygonsIntersectC(poly1Verts, numPoly1Verts, poly2Verts, numPoly2Verts, epsilon, *touches) int
_lib.PolygonsIntersectC.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # poly1Vertices
    ctypes.c_int,  # numPoly1Vertices
    ctypes.POINTER(ctypes.c_double),  # poly2Vertices
    ctypes.c_int,  # numPoly2Vertices
    ctypes.c_double,  # epsilon
    ctypes.POINTER(ctypes.c_int),  # touches
]
_lib.PolygonsIntersectC.restype = ctypes.c_int

# RotateAndTranslatePolygonC(vertices, numVertices, angleRad, dx, dy, output)
_lib.RotateAndTranslatePolygonC.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # vertices
    ctypes.c_int,  # numVertices
    ctypes.c_double,  # angleRad
    ctypes.c_double,  # dx
    ctypes.c_double,  # dy
    ctypes.POINTER(ctypes.c_double),  # output
]
_lib.RotateAndTranslatePolygonC.restype = None

# GetBoundingBoxC(vertices, numVertices, minX, minY, maxX, maxY)
_lib.GetBoundingBoxC.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # vertices
    ctypes.c_int,  # numVertices
    ctypes.POINTER(ctypes.c_double),  # minX
    ctypes.POINTER(ctypes.c_double),  # minY
    ctypes.POINTER(ctypes.c_double),  # maxX
    ctypes.POINTER(ctypes.c_double),  # maxY
]
_lib.GetBoundingBoxC.restype = None

# BatchCollisionCheckC(testVerts, testVertCounts, numTestPolygons, 
#                      placedVerts, placedVertCounts, numPlacedPolygons, 
#                      epsilon, results)
_lib.BatchCollisionCheckC.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # testVertices
    ctypes.POINTER(ctypes.c_int),  # testVertexCounts
    ctypes.c_int,  # numTestPolygons
    ctypes.POINTER(ctypes.c_double),  # placedVertices
    ctypes.POINTER(ctypes.c_int),  # placedVertexCounts
    ctypes.c_int,  # numPlacedPolygons
    ctypes.c_double,  # epsilon
    ctypes.POINTER(ctypes.c_int),  # results
]
_lib.BatchCollisionCheckC.restype = None


def point_in_polygon_go(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Check if a point is inside a polygon using Go implementation.
    
    Args:
        point: 1D array of shape (2,) representing [x, y]
        polygon: 2D array of shape (N, 2) representing polygon vertices
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    point = np.asarray(point, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("Polygon must be an Nx2 array")
    
    # Flatten polygon to 1D array
    polygon_flat = polygon.ravel()
    num_vertices = len(polygon)
    
    result = _lib.PointInPolygonC(
        point[0], point[1],
        polygon_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        num_vertices
    )
    
    return result == 1


def segments_intersect_go(seg1_start: np.ndarray, seg1_end: np.ndarray,
                          seg2_start: np.ndarray, seg2_end: np.ndarray) -> bool:
    """
    Check if two line segments intersect using Go implementation.
    
    Args:
        seg1_start: Start point of segment 1 [x, y]
        seg1_end: End point of segment 1 [x, y]
        seg2_start: Start point of segment 2 [x, y]
        seg2_end: End point of segment 2 [x, y]
        
    Returns:
        True if segments intersect, False otherwise
    """
    seg1_start = np.asarray(seg1_start, dtype=np.float64)
    seg1_end = np.asarray(seg1_end, dtype=np.float64)
    seg2_start = np.asarray(seg2_start, dtype=np.float64)
    seg2_end = np.asarray(seg2_end, dtype=np.float64)
    
    result = _lib.SegmentsIntersectC(
        seg1_start[0], seg1_start[1],
        seg1_end[0], seg1_end[1],
        seg2_start[0], seg2_start[1],
        seg2_end[0], seg2_end[1]
    )
    
    return result == 1


def polygons_intersect_go(poly1: np.ndarray, poly2: np.ndarray, 
                          epsilon: float = 1e-6) -> Tuple[bool, bool]:
    """
    Check if two polygons intersect using Go implementation.
    
    Args:
        poly1: 2D array of shape (N, 2) representing first polygon vertices
        poly2: 2D array of shape (M, 2) representing second polygon vertices
        epsilon: Tolerance for floating point comparisons
        
    Returns:
        Tuple of (intersects, touches):
            - intersects: True if polygons overlap
            - touches: True if polygons only touch at edges/vertices
    """
    poly1 = np.asarray(poly1, dtype=np.float64)
    poly2 = np.asarray(poly2, dtype=np.float64)
    
    if poly1.ndim != 2 or poly1.shape[1] != 2:
        raise ValueError("Polygon 1 must be an Nx2 array")
    if poly2.ndim != 2 or poly2.shape[1] != 2:
        raise ValueError("Polygon 2 must be an Mx2 array")
    
    poly1_flat = poly1.ravel()
    poly2_flat = poly2.ravel()
    touches = ctypes.c_int(0)
    
    result = _lib.PolygonsIntersectC(
        poly1_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(poly1),
        poly2_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(poly2),
        epsilon,
        ctypes.byref(touches)
    )
    
    return result == 1, touches.value == 1


def rotate_and_translate_polygon_go(polygon: np.ndarray, angle_rad: float,
                                    dx: float, dy: float) -> np.ndarray:
    """
    Rotate and translate a polygon using Go implementation.
    
    Args:
        polygon: 2D array of shape (N, 2) representing polygon vertices
        angle_rad: Rotation angle in radians
        dx: Translation in x direction
        dy: Translation in y direction
        
    Returns:
        Transformed polygon as Nx2 array
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("Polygon must be an Nx2 array")
    
    polygon_flat = polygon.ravel()
    output = np.zeros_like(polygon_flat)
    
    _lib.RotateAndTranslatePolygonC(
        polygon_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(polygon),
        angle_rad,
        dx, dy,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    return output.reshape(-1, 2)


def get_bounding_box_go(polygon: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Get the bounding box of a polygon using Go implementation.
    
    Args:
        polygon: 2D array of shape (N, 2) representing polygon vertices
        
    Returns:
        Tuple of (minX, minY, maxX, maxY)
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("Polygon must be an Nx2 array")
    
    polygon_flat = polygon.ravel()
    minX = ctypes.c_double()
    minY = ctypes.c_double()
    maxX = ctypes.c_double()
    maxY = ctypes.c_double()
    
    _lib.GetBoundingBoxC(
        polygon_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(polygon),
        ctypes.byref(minX),
        ctypes.byref(minY),
        ctypes.byref(maxX),
        ctypes.byref(maxY)
    )
    
    return minX.value, minY.value, maxX.value, maxY.value


def batch_collision_check_go(test_polygons: List[np.ndarray],
                             placed_polygons: List[np.ndarray],
                             epsilon: float = 1e-6) -> np.ndarray:
    """
    Check collision for multiple test polygons against placed polygons using Go.
    
    This function uses parallel processing in Go for improved performance.
    
    Args:
        test_polygons: List of test polygons (each as Nx2 array)
        placed_polygons: List of placed polygons (each as Mx2 array)
        epsilon: Tolerance for floating point comparisons
        
    Returns:
        Boolean array indicating which test polygons have collisions
    """
    if not test_polygons:
        return np.array([], dtype=bool)
    
    # Flatten test polygons
    test_vertices = []
    test_counts = []
    for poly in test_polygons:
        poly = np.asarray(poly, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[1] != 2:
            raise ValueError("Each polygon must be an Nx2 array")
        test_vertices.extend(poly.ravel())
        test_counts.append(len(poly))
    
    # Flatten placed polygons
    placed_vertices = []
    placed_counts = []
    for poly in placed_polygons:
        poly = np.asarray(poly, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[1] != 2:
            raise ValueError("Each polygon must be an Nx2 array")
        placed_vertices.extend(poly.ravel())
        placed_counts.append(len(poly))
    
    # Convert to numpy arrays
    test_verts_array = np.array(test_vertices, dtype=np.float64)
    test_counts_array = np.array(test_counts, dtype=np.int32)
    placed_verts_array = np.array(placed_vertices, dtype=np.float64)
    placed_counts_array = np.array(placed_counts, dtype=np.int32)
    
    # Allocate results array
    results = np.zeros(len(test_polygons), dtype=np.int32)
    
    _lib.BatchCollisionCheckC(
        test_verts_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        test_counts_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(test_polygons),
        placed_verts_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        placed_counts_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(placed_polygons),
        epsilon,
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    return results.astype(bool)


# Convenience functions matching the existing API

def point_in_polygon_cpu(points, polygon_vertices):
    """
    CPU fallback for point-in-polygon test using Go.
    Compatible with existing API.
    """
    points = np.asarray(points, dtype=np.float32)
    polygon_vertices = np.asarray(polygon_vertices, dtype=np.float32)
    
    results = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        results[i] = point_in_polygon_go(point, polygon_vertices)
    
    return results


def polygons_intersect_cpu(poly1_verts, poly2_verts, epsilon=1e-6):
    """
    CPU fallback for polygon intersection using Go.
    Compatible with existing API.
    """
    poly1_verts = np.asarray(poly1_verts, dtype=np.float32)
    poly2_verts = np.asarray(poly2_verts, dtype=np.float32)
    
    intersects, touches = polygons_intersect_go(poly1_verts, poly2_verts, epsilon)
    return intersects, touches


def batch_polygon_intersections_cpu(test_polygons, placed_polygons, epsilon=1e-6):
    """
    Batch polygon intersection check using Go.
    Compatible with existing API.
    """
    return batch_collision_check_go(test_polygons, placed_polygons, epsilon)
