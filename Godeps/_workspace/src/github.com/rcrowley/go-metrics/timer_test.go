package metrics

import (
	"math"
	"testing"
	"time"
)

func BenchmarkTimer(b *testing.B) {
	tm := NewTimer()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tm.Update(1)
	}
}

func TestGetOrRegisterTimer(t *testing.T) {
	r := NewRegistry()
	NewRegisteredTimer("foo", r).Update(47)
	if tm := GetOrRegisterTimer("foo", r); 1 != tm.Count() {
		t.Fatal(tm)
	}
}

func TestTimerExtremes(t *testing.T) {
	tm := NewTimer()
	tm.Update(math.MaxInt64)
	tm.Update(0)
	if stdDev := tm.StdDev(); 4.611686018427388e+18 != stdDev {
		t.Errorf("tm.StdDev(): 4.611686018427388e+18 != %v\n", stdDev)
	}
}

func TestTimerFunc(t *testing.T) {
	tm := NewTimer()
	tm.Time(func() { time.Sleep(50e6) })
	if max := tm.Max(); 45e6 > max || max > 55e6 {
		t.Errorf("tm.Max(): 45e6 > %v || %v > 55e6\n", max, max)
	}
}

func TestTimerZero(t *testing.T) {
	tm := NewTimer()
	if count := tm.Count(); 0 != count {
		t.Errorf("tm.Count(): 0 != %v\n", count)
	}
	if min := tm.Min(); 0 != min {
		t.Errorf("tm.Min(): 0 != %v\n", min)
	}
	if max := tm.Max(); 0 != max {
		t.Errorf("tm.Max(): 0 != %v\n", max)
	}
	if mean := tm.Mean(); 0.0 != mean {
		t.Errorf("tm.Mean(): 0.0 != %v\n", mean)
	}
	if stdDev := tm.StdDev(); 0.0 != stdDev {
		t.Errorf("tm.StdDev(): 0.0 != %v\n", stdDev)
	}
	ps := tm.Percentiles([]float64{0.5, 0.75, 0.99})
	if 0.0 != ps[0] {
		t.Errorf("median: 0.0 != %v\n", ps[0])
	}
	if 0.0 != ps[1] {
		t.Errorf("75th percentile: 0.0 != %v\n", ps[1])
	}
	if 0.0 != ps[2] {
		t.Errorf("99th percentile: 0.0 != %v\n", ps[2])
	}
	if rate1 := tm.Rate1(); 0.0 != rate1 {
		t.Errorf("tm.Rate1(): 0.0 != %v\n", rate1)
	}
	if rate5 := tm.Rate5(); 0.0 != rate5 {
		t.Errorf("tm.Rate5(): 0.0 != %v\n", rate5)
	}
	if rate15 := tm.Rate15(); 0.0 != rate15 {
		t.Errorf("tm.Rate15(): 0.0 != %v\n", rate15)
	}
	if rateMean := tm.RateMean(); 0.0 != rateMean {
		t.Errorf("tm.RateMean(): 0.0 != %v\n", rateMean)
	}
}
