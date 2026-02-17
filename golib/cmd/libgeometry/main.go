// Package main provides a C-compatible API for Python interoperability
package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"

	"github.com/leonardosor/kaggle_santa/golib"
)

// C-compatible types for data transfer
type CPoint struct {
	X, Y float64
}

// PointInPolygonC checks if a point is inside a polygon
// Returns 1 if inside, 0 if outside
//
//export PointInPolygonC
func PointInPolygonC(pointX, pointY float64, vertices *float64, numVertices C.int) C.int {
	// Convert C arrays to Go slices
	vertexCount := int(numVertices)
	verticesSlice := (*[1 << 30]float64)(unsafe.Pointer(vertices))[:vertexCount*2:vertexCount*2]

	polygon := make([]golib.Point, vertexCount)
	for i := 0; i < vertexCount; i++ {
		polygon[i] = golib.Point{
			X: verticesSlice[i*2],
			Y: verticesSlice[i*2+1],
		}
	}

	point := golib.Point{X: pointX, Y: pointY}
	if golib.PointInPolygon(point, polygon) {
		return 1
	}
	return 0
}

// SegmentsIntersectC checks if two line segments intersect
// Returns 1 if they intersect, 0 otherwise
//
//export SegmentsIntersectC
func SegmentsIntersectC(s1StartX, s1StartY, s1EndX, s1EndY, s2StartX, s2StartY, s2EndX, s2EndY float64) C.int {
	s1Start := golib.Point{X: s1StartX, Y: s1StartY}
	s1End := golib.Point{X: s1EndX, Y: s1EndY}
	s2Start := golib.Point{X: s2StartX, Y: s2StartY}
	s2End := golib.Point{X: s2EndX, Y: s2EndY}

	if golib.SegmentsIntersect(s1Start, s1End, s2Start, s2End) {
		return 1
	}
	return 0
}

// PolygonsIntersectC checks if two polygons intersect
// Returns 1 if they intersect (overlap), 0 otherwise
// Second return value (through touches pointer) indicates if they only touch
//
//export PolygonsIntersectC
func PolygonsIntersectC(
	poly1Vertices *float64, numPoly1Vertices C.int,
	poly2Vertices *float64, numPoly2Vertices C.int,
	epsilon float64,
	touches *C.int,
) C.int {
	// Convert polygon 1
	count1 := int(numPoly1Vertices)
	verts1Slice := (*[1 << 30]float64)(unsafe.Pointer(poly1Vertices))[:count1*2:count1*2]
	poly1 := make([]golib.Point, count1)
	for i := 0; i < count1; i++ {
		poly1[i] = golib.Point{
			X: verts1Slice[i*2],
			Y: verts1Slice[i*2+1],
		}
	}

	// Convert polygon 2
	count2 := int(numPoly2Vertices)
	verts2Slice := (*[1 << 30]float64)(unsafe.Pointer(poly2Vertices))[:count2*2:count2*2]
	poly2 := make([]golib.Point, count2)
	for i := 0; i < count2; i++ {
		poly2[i] = golib.Point{
			X: verts2Slice[i*2],
			Y: verts2Slice[i*2+1],
		}
	}

	intersects, touchesResult := golib.PolygonsIntersect(poly1, poly2, epsilon)

	if touches != nil {
		if touchesResult {
			*touches = 1
		} else {
			*touches = 0
		}
	}

	if intersects {
		return 1
	}
	return 0
}

// RotateAndTranslatePolygonC applies rotation and translation to a polygon
// Results are written to the output array (must be pre-allocated)
//
//export RotateAndTranslatePolygonC
func RotateAndTranslatePolygonC(
	vertices *float64, numVertices C.int,
	angleRad, dx, dy float64,
	output *float64,
) {
	// Convert input polygon
	count := int(numVertices)
	vertsSlice := (*[1 << 30]float64)(unsafe.Pointer(vertices))[:count*2:count*2]
	polygon := make([]golib.Point, count)
	for i := 0; i < count; i++ {
		polygon[i] = golib.Point{
			X: vertsSlice[i*2],
			Y: vertsSlice[i*2+1],
		}
	}

	// Transform
	result := golib.RotateAndTranslatePolygon(polygon, angleRad, dx, dy)

	// Write to output
	outputSlice := (*[1 << 30]float64)(unsafe.Pointer(output))[:count*2:count*2]
	for i, p := range result {
		outputSlice[i*2] = p.X
		outputSlice[i*2+1] = p.Y
	}
}

// GetBoundingBoxC computes the bounding box of a polygon
// Returns values through output pointers
//
//export GetBoundingBoxC
func GetBoundingBoxC(
	vertices *float64, numVertices C.int,
	minX, minY, maxX, maxY *float64,
) {
	// Convert polygon
	count := int(numVertices)
	vertsSlice := (*[1 << 30]float64)(unsafe.Pointer(vertices))[:count*2:count*2]
	polygon := make([]golib.Point, count)
	for i := 0; i < count; i++ {
		polygon[i] = golib.Point{
			X: vertsSlice[i*2],
			Y: vertsSlice[i*2+1],
		}
	}

	bbox := golib.GetBoundingBox(polygon)

	*minX = bbox.MinX
	*minY = bbox.MinY
	*maxX = bbox.MaxX
	*maxY = bbox.MaxY
}

// BatchCollisionCheckC checks collision for multiple test polygons against placed polygons
// Results are written to the output array (1 = collision, 0 = no collision)
//
//export BatchCollisionCheckC
func BatchCollisionCheckC(
	// Test polygons: flat array of all vertices, then array of vertex counts
	testVertices *float64, testVertexCounts *C.int, numTestPolygons C.int,
	// Placed polygons: flat array of all vertices, then array of vertex counts
	placedVertices *float64, placedVertexCounts *C.int, numPlacedPolygons C.int,
	epsilon float64,
	// Output: array of results
	results *C.int,
) {
	numTests := int(numTestPolygons)
	numPlaced := int(numPlacedPolygons)

	// Convert test vertex counts
	testCountsSlice := (*[1 << 30]C.int)(unsafe.Pointer(testVertexCounts))[:numTests:numTests]

	// Convert placed vertex counts
	placedCountsSlice := (*[1 << 30]C.int)(unsafe.Pointer(placedVertexCounts))[:numPlaced:numPlaced]

	// Parse test polygons
	testPolygons := make([][]golib.Point, numTests)
	testVertOffset := 0
	testVertsArray := (*[1 << 30]float64)(unsafe.Pointer(testVertices))

	for i := 0; i < numTests; i++ {
		count := int(testCountsSlice[i])
		poly := make([]golib.Point, count)
		for j := 0; j < count; j++ {
			idx := testVertOffset + j*2
			poly[j] = golib.Point{
				X: testVertsArray[idx],
				Y: testVertsArray[idx+1],
			}
		}
		testPolygons[i] = poly
		testVertOffset += count * 2
	}

	// Parse placed polygons
	placedPolygons := make([][]golib.Point, numPlaced)
	placedVertOffset := 0
	placedVertsArray := (*[1 << 30]float64)(unsafe.Pointer(placedVertices))

	for i := 0; i < numPlaced; i++ {
		count := int(placedCountsSlice[i])
		poly := make([]golib.Point, count)
		for j := 0; j < count; j++ {
			idx := placedVertOffset + j*2
			poly[j] = golib.Point{
				X: placedVertsArray[idx],
				Y: placedVertsArray[idx+1],
			}
		}
		placedPolygons[i] = poly
		placedVertOffset += count * 2
	}

	// Perform batch collision check
	collisionResults := golib.BatchCollisionCheck(testPolygons, placedPolygons, epsilon)

	// Write results
	resultsSlice := (*[1 << 30]C.int)(unsafe.Pointer(results))[:numTests:numTests]
	for i, hasCollision := range collisionResults {
		if hasCollision {
			resultsSlice[i] = 1
		} else {
			resultsSlice[i] = 0
		}
	}
}

func main() {
	// Required for building as a shared library
}
