package histogram

import (
	"math/rand"
	"testing"
)

func BenchmarkInsert10Bins(b *testing.B) {
	b.StopTimer()
	h := New(10)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		f := rand.ExpFloat64()
		h.Insert(f)
	}
}

func BenchmarkInsert100Bins(b *testing.B) {
	b.StopTimer()
	h := New(100)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		f := rand.ExpFloat64()
		h.Insert(f)
	}
}
