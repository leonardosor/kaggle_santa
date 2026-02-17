// Package golib provides high-performance geometric operations for polygon collision detection
package golib

import (
	"math"
	"sync"
)

// Point represents a 2D point with x and y coordinates
type Point struct {
	X, Y float64
}

// Polygon represents a polygon as a slice of vertices
type Polygon struct {
	Vertices []Point
}

// BoundingBox represents an axis-aligned bounding box
type BoundingBox struct {
	MinX, MinY, MaxX, MaxY float64
}

// PointInPolygon checks if a point is inside a polygon using ray casting algorithm
// This is the core algorithm used millions of times in collision detection
func PointInPolygon(point Point, polygon []Point) bool {
	if len(polygon) < 3 {
		return false
	}

	inside := false
	n := len(polygon)

	for i := 0; i < n; i++ {
		v1 := polygon[i]
		v2 := polygon[(i+1)%n]

		// Check if ray from point crosses edge (v1, v2)
		// Edge must span the point's y coordinate
		if (v1.Y > point.Y) != (v2.Y > point.Y) {
			// Calculate x coordinate of intersection
			xIntersect := (v2.X-v1.X)*(point.Y-v1.Y)/(v2.Y-v1.Y) + v1.X

			// Ray crosses edge if intersection is to the right of point
			if point.X < xIntersect {
				inside = !inside
			}
		}
	}

	return inside
}

// SegmentsIntersect checks if two line segments intersect using cross product method
func SegmentsIntersect(s1Start, s1End, s2Start, s2End Point) bool {
	// Direction vectors
	d1 := Point{s1End.X - s1Start.X, s1End.Y - s1Start.Y}
	d2 := Point{s2End.X - s2Start.X, s2End.Y - s2Start.Y}

	// Cross products to check orientation
	cross1 := crossProduct(d1, Point{s2Start.X - s1Start.X, s2Start.Y - s1Start.Y})
	cross2 := crossProduct(d1, Point{s2End.X - s1Start.X, s2End.Y - s1Start.Y})
	cross3 := crossProduct(d2, Point{s1Start.X - s2Start.X, s1Start.Y - s2Start.Y})
	cross4 := crossProduct(d2, Point{s1End.X - s2Start.X, s1End.Y - s2Start.Y})

	// Segments intersect if points are on opposite sides
	if cross1*cross2 < 0 && cross3*cross4 < 0 {
		return true
	}

	return false
}

// crossProduct computes 2D cross product (z-component of 3D cross product)
func crossProduct(v1, v2 Point) float64 {
	return v1.X*v2.Y - v1.Y*v2.X
}

// PolygonsIntersect checks if two polygons intersect
// Returns (intersects, touches) - intersects means actual overlap, touches means only edges touch
func PolygonsIntersect(poly1, poly2 []Point, epsilon float64) (bool, bool) {
	if len(poly1) < 3 || len(poly2) < 3 {
		return false, false
	}

	// First check if bounding boxes overlap
	bbox1 := computeBoundingBox(poly1)
	bbox2 := computeBoundingBox(poly2)

	if !boundingBoxesOverlap(bbox1, bbox2, epsilon) {
		return false, false
	}

	// Check if any vertex of poly1 is inside poly2
	for _, v := range poly1 {
		if PointInPolygon(v, poly2) {
			return true, false
		}
	}

	// Check if any vertex of poly2 is inside poly1
	for _, v := range poly2 {
		if PointInPolygon(v, poly1) {
			return true, false
		}
	}

	// Check if any edges intersect
	for i := 0; i < len(poly1); i++ {
		s1Start := poly1[i]
		s1End := poly1[(i+1)%len(poly1)]

		for j := 0; j < len(poly2); j++ {
			s2Start := poly2[j]
			s2End := poly2[(j+1)%len(poly2)]

			if SegmentsIntersect(s1Start, s1End, s2Start, s2End) {
				return true, false
			}
		}
	}

	return false, false
}

// computeBoundingBox calculates the axis-aligned bounding box of a polygon
func computeBoundingBox(polygon []Point) BoundingBox {
	if len(polygon) == 0 {
		return BoundingBox{}
	}

	bbox := BoundingBox{
		MinX: polygon[0].X,
		MinY: polygon[0].Y,
		MaxX: polygon[0].X,
		MaxY: polygon[0].Y,
	}

	for _, p := range polygon[1:] {
		if p.X < bbox.MinX {
			bbox.MinX = p.X
		}
		if p.X > bbox.MaxX {
			bbox.MaxX = p.X
		}
		if p.Y < bbox.MinY {
			bbox.MinY = p.Y
		}
		if p.Y > bbox.MaxY {
			bbox.MaxY = p.Y
		}
	}

	return bbox
}

// boundingBoxesOverlap checks if two bounding boxes overlap
func boundingBoxesOverlap(bbox1, bbox2 BoundingBox, epsilon float64) bool {
	return !(bbox1.MaxX+epsilon < bbox2.MinX ||
		bbox2.MaxX+epsilon < bbox1.MinX ||
		bbox1.MaxY+epsilon < bbox2.MinY ||
		bbox2.MaxY+epsilon < bbox1.MinY)
}

// RotatePoint rotates a point around origin by angle in radians
func RotatePoint(p Point, angleRad float64) Point {
	cos := math.Cos(angleRad)
	sin := math.Sin(angleRad)

	return Point{
		X: p.X*cos - p.Y*sin,
		Y: p.X*sin + p.Y*cos,
	}
}

// TranslatePolygon translates all vertices of a polygon by dx, dy
func TranslatePolygon(polygon []Point, dx, dy float64) []Point {
	result := make([]Point, len(polygon))
	for i, p := range polygon {
		result[i] = Point{X: p.X + dx, Y: p.Y + dy}
	}
	return result
}

// RotatePolygon rotates all vertices of a polygon around origin by angle in radians
func RotatePolygon(polygon []Point, angleRad float64) []Point {
	result := make([]Point, len(polygon))
	for i, p := range polygon {
		result[i] = RotatePoint(p, angleRad)
	}
	return result
}

// RotateAndTranslatePolygon applies both rotation and translation to a polygon
func RotateAndTranslatePolygon(polygon []Point, angleRad, dx, dy float64) []Point {
	result := make([]Point, len(polygon))
	cos := math.Cos(angleRad)
	sin := math.Sin(angleRad)

	for i, p := range polygon {
		// Rotate
		rotX := p.X*cos - p.Y*sin
		rotY := p.X*sin + p.Y*cos
		// Translate
		result[i] = Point{X: rotX + dx, Y: rotY + dy}
	}
	return result
}

// BatchCollisionCheck checks collision between multiple test polygons and placed polygons
// Returns a boolean slice indicating which test polygons collide
// This function uses goroutines for parallel processing
func BatchCollisionCheck(testPolygons [][]Point, placedPolygons [][]Point, epsilon float64) []bool {
	numTests := len(testPolygons)
	results := make([]bool, numTests)

	// Use worker pool for parallel processing
	numWorkers := 8 // Can be tuned based on CPU cores
	if numTests < numWorkers {
		numWorkers = numTests
	}

	var wg sync.WaitGroup
	jobs := make(chan int, numTests)

	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for testIdx := range jobs {
				testPoly := testPolygons[testIdx]
				hasCollision := false

				// Check against all placed polygons
				for _, placedPoly := range placedPolygons {
					intersects, _ := PolygonsIntersect(testPoly, placedPoly, epsilon)
					if intersects {
						hasCollision = true
						break
					}
				}

				results[testIdx] = hasCollision
			}
		}()
	}

	// Send jobs
	for i := 0; i < numTests; i++ {
		jobs <- i
	}
	close(jobs)

	// Wait for completion
	wg.Wait()

	return results
}

// GetBoundingBox returns the bounding box of a polygon
func GetBoundingBox(polygon []Point) BoundingBox {
	return computeBoundingBox(polygon)
}

// ComputeUnionBoundingBox computes the bounding box of multiple polygons
func ComputeUnionBoundingBox(polygons [][]Point) BoundingBox {
	if len(polygons) == 0 {
		return BoundingBox{}
	}

	bbox := computeBoundingBox(polygons[0])

	for _, poly := range polygons[1:] {
		polyBbox := computeBoundingBox(poly)

		if polyBbox.MinX < bbox.MinX {
			bbox.MinX = polyBbox.MinX
		}
		if polyBbox.MaxX > bbox.MaxX {
			bbox.MaxX = polyBbox.MaxX
		}
		if polyBbox.MinY < bbox.MinY {
			bbox.MinY = polyBbox.MinY
		}
		if polyBbox.MaxY > bbox.MaxY {
			bbox.MaxY = polyBbox.MaxY
		}
	}

	return bbox
}
