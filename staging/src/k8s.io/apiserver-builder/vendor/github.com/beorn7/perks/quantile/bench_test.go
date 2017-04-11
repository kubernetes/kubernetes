package quantile

import (
	"testing"
)

func BenchmarkInsertTargeted(b *testing.B) {
	b.ReportAllocs()

	s := NewTargeted(Targets)
	b.ResetTimer()
	for i := float64(0); i < float64(b.N); i++ {
		s.Insert(i)
	}
}

func BenchmarkInsertTargetedSmallEpsilon(b *testing.B) {
	s := NewTargeted(TargetsSmallEpsilon)
	b.ResetTimer()
	for i := float64(0); i < float64(b.N); i++ {
		s.Insert(i)
	}
}

func BenchmarkInsertBiased(b *testing.B) {
	s := NewLowBiased(0.01)
	b.ResetTimer()
	for i := float64(0); i < float64(b.N); i++ {
		s.Insert(i)
	}
}

func BenchmarkInsertBiasedSmallEpsilon(b *testing.B) {
	s := NewLowBiased(0.0001)
	b.ResetTimer()
	for i := float64(0); i < float64(b.N); i++ {
		s.Insert(i)
	}
}

func BenchmarkQuery(b *testing.B) {
	s := NewTargeted(Targets)
	for i := float64(0); i < 1e6; i++ {
		s.Insert(i)
	}
	b.ResetTimer()
	n := float64(b.N)
	for i := float64(0); i < n; i++ {
		s.Query(i / n)
	}
}

func BenchmarkQuerySmallEpsilon(b *testing.B) {
	s := NewTargeted(TargetsSmallEpsilon)
	for i := float64(0); i < 1e6; i++ {
		s.Insert(i)
	}
	b.ResetTimer()
	n := float64(b.N)
	for i := float64(0); i < n; i++ {
		s.Query(i / n)
	}
}
