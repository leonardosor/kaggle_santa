package golib

import (
	"math"
	"testing"
)

func TestPointInPolygon(t *testing.T) {
	// Square polygon
	square := []Point{
		{0, 0},
		{10, 0},
		{10, 10},
		{0, 10},
	}

	tests := []struct {
		name     string
		point    Point
		expected bool
	}{
		{"Inside center", Point{5, 5}, true},
		{"Outside", Point{15, 15}, false},
		{"Outside negative", Point{-5, 5}, false},
		{"On vertex (edge case)", Point{0, 0}, true}, // Ray casting implementation detail
		{"On edge (edge case)", Point{5, 0}, true},    // Ray casting implementation detail
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := PointInPolygon(tt.point, square)
			if result != tt.expected {
				t.Errorf("PointInPolygon(%v) = %v, want %v", tt.point, result, tt.expected)
			}
		})
	}
}

func TestSegmentsIntersect(t *testing.T) {
	tests := []struct {
		name     string
		s1Start  Point
		s1End    Point
		s2Start  Point
		s2End    Point
		expected bool
	}{
		{
			"Crossing segments",
			Point{0, 0}, Point{10, 10},
			Point{0, 10}, Point{10, 0},
			true,
		},
		{
			"Parallel segments",
			Point{0, 0}, Point{10, 0},
			Point{0, 5}, Point{10, 5},
			false,
		},
		{
			"Non-intersecting",
			Point{0, 0}, Point{5, 5},
			Point{10, 10}, Point{15, 15},
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SegmentsIntersect(tt.s1Start, tt.s1End, tt.s2Start, tt.s2End)
			if result != tt.expected {
				t.Errorf("SegmentsIntersect() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestPolygonsIntersect(t *testing.T) {
	// Square at origin
	poly1 := []Point{
		{0, 0},
		{10, 0},
		{10, 10},
		{0, 10},
	}

	tests := []struct {
		name              string
		poly2             []Point
		expectedIntersect bool
	}{
		{
			"Overlapping square",
			[]Point{
				{5, 5},
				{15, 5},
				{15, 15},
				{5, 15},
			},
			true,
		},
		{
			"Non-overlapping square",
			[]Point{
				{20, 20},
				{30, 20},
				{30, 30},
				{20, 30},
			},
			false,
		},
		{
			"Contained square",
			[]Point{
				{2, 2},
				{8, 2},
				{8, 8},
				{2, 8},
			},
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			intersects, _ := PolygonsIntersect(poly1, tt.poly2, 1e-6)
			if intersects != tt.expectedIntersect {
				t.Errorf("PolygonsIntersect() = %v, want %v", intersects, tt.expectedIntersect)
			}
		})
	}
}

func TestRotatePoint(t *testing.T) {
	p := Point{1, 0}

	// Rotate 90 degrees (π/2 radians)
	rotated := RotatePoint(p, math.Pi/2)

	// Should be approximately (0, 1)
	if math.Abs(rotated.X) > 1e-10 || math.Abs(rotated.Y-1) > 1e-10 {
		t.Errorf("RotatePoint(Point{1,0}, π/2) = %v, want approximately {0, 1}", rotated)
	}
}

func TestRotatePolygon(t *testing.T) {
	square := []Point{
		{1, 0},
		{0, 1},
		{-1, 0},
		{0, -1},
	}

	// Rotate 90 degrees
	rotated := RotatePolygon(square, math.Pi/2)

	// First vertex should be approximately (0, 1)
	if math.Abs(rotated[0].X) > 1e-10 || math.Abs(rotated[0].Y-1) > 1e-10 {
		t.Errorf("First vertex after rotation = %v, want approximately {0, 1}", rotated[0])
	}
}

func TestTranslatePolygon(t *testing.T) {
	square := []Point{
		{0, 0},
		{10, 0},
		{10, 10},
		{0, 10},
	}

	translated := TranslatePolygon(square, 5, 5)

	expected := []Point{
		{5, 5},
		{15, 5},
		{15, 15},
		{5, 15},
	}

	for i, p := range translated {
		if p.X != expected[i].X || p.Y != expected[i].Y {
			t.Errorf("Vertex %d: got %v, want %v", i, p, expected[i])
		}
	}
}

func TestRotateAndTranslatePolygon(t *testing.T) {
	square := []Point{
		{1, 0},
		{0, 1},
		{-1, 0},
		{0, -1},
	}

	// Rotate 90 degrees and translate by (10, 10)
	result := RotateAndTranslatePolygon(square, math.Pi/2, 10, 10)

	// First vertex: rotated (1, 0) -> (0, 1) then translated -> (10, 11)
	if math.Abs(result[0].X-10) > 1e-10 || math.Abs(result[0].Y-11) > 1e-10 {
		t.Errorf("First vertex = %v, want approximately {10, 11}", result[0])
	}
}

func TestComputeBoundingBox(t *testing.T) {
	polygon := []Point{
		{0, 0},
		{10, 5},
		{5, 10},
		{-2, 3},
	}

	bbox := computeBoundingBox(polygon)

	if bbox.MinX != -2 || bbox.MinY != 0 || bbox.MaxX != 10 || bbox.MaxY != 10 {
		t.Errorf("BoundingBox = %+v, want MinX=-2, MinY=0, MaxX=10, MaxY=10", bbox)
	}
}

func TestBoundingBoxesOverlap(t *testing.T) {
	bbox1 := BoundingBox{MinX: 0, MinY: 0, MaxX: 10, MaxY: 10}

	tests := []struct {
		name     string
		bbox2    BoundingBox
		expected bool
	}{
		{
			"Overlapping",
			BoundingBox{MinX: 5, MinY: 5, MaxX: 15, MaxY: 15},
			true,
		},
		{
			"Non-overlapping",
			BoundingBox{MinX: 20, MinY: 20, MaxX: 30, MaxY: 30},
			false,
		},
		{
			"Touching edge",
			BoundingBox{MinX: 10, MinY: 0, MaxX: 20, MaxY: 10},
			true, // Touching edges are considered overlapping with <= comparison
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := boundingBoxesOverlap(bbox1, tt.bbox2, 0)
			if result != tt.expected {
				t.Errorf("boundingBoxesOverlap() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBatchCollisionCheck(t *testing.T) {
	// Placed polygon
	placed := [][]Point{
		{
			{0, 0},
			{10, 0},
			{10, 10},
			{0, 10},
		},
	}

	// Test polygons: one overlapping, one not
	testPolys := [][]Point{
		{ // Overlapping
			{5, 5},
			{15, 5},
			{15, 15},
			{5, 15},
		},
		{ // Not overlapping
			{20, 20},
			{30, 20},
			{30, 30},
			{20, 30},
		},
	}

	results := BatchCollisionCheck(testPolys, placed, 1e-6)

	if len(results) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(results))
	}

	if !results[0] {
		t.Error("First polygon should collide")
	}

	if results[1] {
		t.Error("Second polygon should not collide")
	}
}

func TestComputeUnionBoundingBox(t *testing.T) {
	polygons := [][]Point{
		{
			{0, 0},
			{10, 0},
			{10, 10},
			{0, 10},
		},
		{
			{15, 15},
			{25, 15},
			{25, 25},
			{15, 25},
		},
	}

	bbox := ComputeUnionBoundingBox(polygons)

	if bbox.MinX != 0 || bbox.MinY != 0 || bbox.MaxX != 25 || bbox.MaxY != 25 {
		t.Errorf("UnionBoundingBox = %+v, want MinX=0, MinY=0, MaxX=25, MaxY=25", bbox)
	}
}

// Benchmark tests
func BenchmarkPointInPolygon(b *testing.B) {
	square := []Point{
		{0, 0},
		{10, 0},
		{10, 10},
		{0, 10},
	}
	point := Point{5, 5}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PointInPolygon(point, square)
	}
}

func BenchmarkPolygonsIntersect(b *testing.B) {
	poly1 := []Point{
		{0, 0},
		{10, 0},
		{10, 10},
		{0, 10},
	}
	poly2 := []Point{
		{5, 5},
		{15, 5},
		{15, 15},
		{5, 15},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PolygonsIntersect(poly1, poly2, 1e-6)
	}
}

func BenchmarkBatchCollisionCheck(b *testing.B) {
	// Create 10 placed polygons
	placed := make([][]Point, 10)
	for i := 0; i < 10; i++ {
		x := float64(i * 15)
		placed[i] = []Point{
			{x, 0},
			{x + 10, 0},
			{x + 10, 10},
			{x, 10},
		}
	}

	// Create 100 test polygons
	testPolys := make([][]Point, 100)
	for i := 0; i < 100; i++ {
		x := float64(i * 2)
		y := float64(i % 10 * 2)
		testPolys[i] = []Point{
			{x, y},
			{x + 5, y},
			{x + 5, y + 5},
			{x, y + 5},
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BatchCollisionCheck(testPolys, placed, 1e-6)
	}
}
