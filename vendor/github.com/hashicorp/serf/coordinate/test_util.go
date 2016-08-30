package coordinate

import (
	"math"
	"testing"
)

// verifyEqualFloats will compare f1 and f2 and fail if they are not
// "equal" within a threshold.
func verifyEqualFloats(t *testing.T, f1 float64, f2 float64) {
	const zeroThreshold = 1.0e-6
	if math.Abs(f1-f2) > zeroThreshold {
		t.Fatalf("equal assertion fail, %9.6f != %9.6f", f1, f2)
	}
}

// verifyEqualVectors will compare vec1 and vec2 and fail if they are not
// "equal" within a threshold.
func verifyEqualVectors(t *testing.T, vec1 []float64, vec2 []float64) {
	if len(vec1) != len(vec2) {
		t.Fatalf("vector length mismatch, %d != %d", len(vec1), len(vec2))
	}

	for i, _ := range vec1 {
		verifyEqualFloats(t, vec1[i], vec2[i])
	}
}
