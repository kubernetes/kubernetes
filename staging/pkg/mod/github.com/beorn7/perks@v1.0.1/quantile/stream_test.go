package quantile

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

var (
	Targets = map[float64]float64{
		0.01: 0.001,
		0.10: 0.01,
		0.50: 0.05,
		0.90: 0.01,
		0.99: 0.001,
	}
	TargetsSmallEpsilon = map[float64]float64{
		0.01: 0.0001,
		0.10: 0.001,
		0.50: 0.005,
		0.90: 0.001,
		0.99: 0.0001,
	}
	LowQuantiles  = []float64{0.01, 0.1, 0.5}
	HighQuantiles = []float64{0.99, 0.9, 0.5}
)

const RelativeEpsilon = 0.01

func verifyPercsWithAbsoluteEpsilon(t *testing.T, a []float64, s *Stream) {
	sort.Float64s(a)
	for quantile, epsilon := range Targets {
		n := float64(len(a))
		k := int(quantile * n)
		if k < 1 {
			k = 1
		}
		lower := int((quantile - epsilon) * n)
		if lower < 1 {
			lower = 1
		}
		upper := int(math.Ceil((quantile + epsilon) * n))
		if upper > len(a) {
			upper = len(a)
		}
		w, min, max := a[k-1], a[lower-1], a[upper-1]
		if g := s.Query(quantile); g < min || g > max {
			t.Errorf("q=%f: want %v [%f,%f], got %v", quantile, w, min, max, g)
		}
	}
}

func verifyLowPercsWithRelativeEpsilon(t *testing.T, a []float64, s *Stream) {
	sort.Float64s(a)
	for _, qu := range LowQuantiles {
		n := float64(len(a))
		k := int(qu * n)

		lowerRank := int((1 - RelativeEpsilon) * qu * n)
		upperRank := int(math.Ceil((1 + RelativeEpsilon) * qu * n))
		w, min, max := a[k-1], a[lowerRank-1], a[upperRank-1]
		if g := s.Query(qu); g < min || g > max {
			t.Errorf("q=%f: want %v [%f,%f], got %v", qu, w, min, max, g)
		}
	}
}

func verifyHighPercsWithRelativeEpsilon(t *testing.T, a []float64, s *Stream) {
	sort.Float64s(a)
	for _, qu := range HighQuantiles {
		n := float64(len(a))
		k := int(qu * n)

		lowerRank := int((1 - (1+RelativeEpsilon)*(1-qu)) * n)
		upperRank := int(math.Ceil((1 - (1-RelativeEpsilon)*(1-qu)) * n))
		w, min, max := a[k-1], a[lowerRank-1], a[upperRank-1]
		if g := s.Query(qu); g < min || g > max {
			t.Errorf("q=%f: want %v [%f,%f], got %v", qu, w, min, max, g)
		}
	}
}

func populateStream(s *Stream) []float64 {
	a := make([]float64, 0, 1e5+100)
	for i := 0; i < cap(a); i++ {
		v := rand.NormFloat64()
		// Add 5% asymmetric outliers.
		if i%20 == 0 {
			v = v*v + 1
		}
		s.Insert(v)
		a = append(a, v)
	}
	return a
}

func TestTargetedQuery(t *testing.T) {
	rand.Seed(42)
	s := NewTargeted(Targets)
	a := populateStream(s)
	verifyPercsWithAbsoluteEpsilon(t, a, s)
}

func TestTargetedQuerySmallSampleSize(t *testing.T) {
	rand.Seed(42)
	s := NewTargeted(TargetsSmallEpsilon)
	a := []float64{1, 2, 3, 4, 5}
	for _, v := range a {
		s.Insert(v)
	}
	verifyPercsWithAbsoluteEpsilon(t, a, s)
	// If not yet flushed, results should be precise:
	if !s.flushed() {
		for φ, want := range map[float64]float64{
			0.01: 1,
			0.10: 1,
			0.50: 3,
			0.90: 5,
			0.99: 5,
		} {
			if got := s.Query(φ); got != want {
				t.Errorf("want %f for φ=%f, got %f", want, φ, got)
			}
		}
	}
}

func TestLowBiasedQuery(t *testing.T) {
	rand.Seed(42)
	s := NewLowBiased(RelativeEpsilon)
	a := populateStream(s)
	verifyLowPercsWithRelativeEpsilon(t, a, s)
}

func TestHighBiasedQuery(t *testing.T) {
	rand.Seed(42)
	s := NewHighBiased(RelativeEpsilon)
	a := populateStream(s)
	verifyHighPercsWithRelativeEpsilon(t, a, s)
}

// BrokenTestTargetedMerge is broken, see Merge doc comment.
func BrokenTestTargetedMerge(t *testing.T) {
	rand.Seed(42)
	s1 := NewTargeted(Targets)
	s2 := NewTargeted(Targets)
	a := populateStream(s1)
	a = append(a, populateStream(s2)...)
	s1.Merge(s2.Samples())
	verifyPercsWithAbsoluteEpsilon(t, a, s1)
}

// BrokenTestLowBiasedMerge is broken, see Merge doc comment.
func BrokenTestLowBiasedMerge(t *testing.T) {
	rand.Seed(42)
	s1 := NewLowBiased(RelativeEpsilon)
	s2 := NewLowBiased(RelativeEpsilon)
	a := populateStream(s1)
	a = append(a, populateStream(s2)...)
	s1.Merge(s2.Samples())
	verifyLowPercsWithRelativeEpsilon(t, a, s2)
}

// BrokenTestHighBiasedMerge is broken, see Merge doc comment.
func BrokenTestHighBiasedMerge(t *testing.T) {
	rand.Seed(42)
	s1 := NewHighBiased(RelativeEpsilon)
	s2 := NewHighBiased(RelativeEpsilon)
	a := populateStream(s1)
	a = append(a, populateStream(s2)...)
	s1.Merge(s2.Samples())
	verifyHighPercsWithRelativeEpsilon(t, a, s2)
}

func TestUncompressed(t *testing.T) {
	q := NewTargeted(Targets)
	for i := 100; i > 0; i-- {
		q.Insert(float64(i))
	}
	if g := q.Count(); g != 100 {
		t.Errorf("want count 100, got %d", g)
	}
	// Before compression, Query should have 100% accuracy.
	for quantile := range Targets {
		w := quantile * 100
		if g := q.Query(quantile); g != w {
			t.Errorf("want %f, got %f", w, g)
		}
	}
}

func TestUncompressedSamples(t *testing.T) {
	q := NewTargeted(map[float64]float64{0.99: 0.001})
	for i := 1; i <= 100; i++ {
		q.Insert(float64(i))
	}
	if g := q.Samples().Len(); g != 100 {
		t.Errorf("want count 100, got %d", g)
	}
}

func TestUncompressedOne(t *testing.T) {
	q := NewTargeted(map[float64]float64{0.99: 0.01})
	q.Insert(3.14)
	if g := q.Query(0.90); g != 3.14 {
		t.Error("want PI, got", g)
	}
}

func TestDefaults(t *testing.T) {
	if g := NewTargeted(map[float64]float64{0.99: 0.001}).Query(0.99); g != 0 {
		t.Errorf("want 0, got %f", g)
	}
}
