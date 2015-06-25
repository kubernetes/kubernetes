package metrics

import (
	"math/rand"
	"runtime"
	"testing"
	"time"
)

// Benchmark{Compute,Copy}{1000,1000000} demonstrate that, even for relatively
// expensive computations like Variance, the cost of copying the Sample, as
// approximated by a make and copy, is much greater than the cost of the
// computation for small samples and only slightly less for large samples.
func BenchmarkCompute1000(b *testing.B) {
	s := make([]int64, 1000)
	for i := 0; i < len(s); i++ {
		s[i] = int64(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SampleVariance(s)
	}
}
func BenchmarkCompute1000000(b *testing.B) {
	s := make([]int64, 1000000)
	for i := 0; i < len(s); i++ {
		s[i] = int64(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SampleVariance(s)
	}
}
func BenchmarkCopy1000(b *testing.B) {
	s := make([]int64, 1000)
	for i := 0; i < len(s); i++ {
		s[i] = int64(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sCopy := make([]int64, len(s))
		copy(sCopy, s)
	}
}
func BenchmarkCopy1000000(b *testing.B) {
	s := make([]int64, 1000000)
	for i := 0; i < len(s); i++ {
		s[i] = int64(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sCopy := make([]int64, len(s))
		copy(sCopy, s)
	}
}

func BenchmarkExpDecaySample257(b *testing.B) {
	benchmarkSample(b, NewExpDecaySample(257, 0.015))
}

func BenchmarkExpDecaySample514(b *testing.B) {
	benchmarkSample(b, NewExpDecaySample(514, 0.015))
}

func BenchmarkExpDecaySample1028(b *testing.B) {
	benchmarkSample(b, NewExpDecaySample(1028, 0.015))
}

func BenchmarkUniformSample257(b *testing.B) {
	benchmarkSample(b, NewUniformSample(257))
}

func BenchmarkUniformSample514(b *testing.B) {
	benchmarkSample(b, NewUniformSample(514))
}

func BenchmarkUniformSample1028(b *testing.B) {
	benchmarkSample(b, NewUniformSample(1028))
}

func TestExpDecaySample10(t *testing.T) {
	rand.Seed(1)
	s := NewExpDecaySample(100, 0.99)
	for i := 0; i < 10; i++ {
		s.Update(int64(i))
	}
	if size := s.Count(); 10 != size {
		t.Errorf("s.Count(): 10 != %v\n", size)
	}
	if size := s.Size(); 10 != size {
		t.Errorf("s.Size(): 10 != %v\n", size)
	}
	if l := len(s.Values()); 10 != l {
		t.Errorf("len(s.Values()): 10 != %v\n", l)
	}
	for _, v := range s.Values() {
		if v > 10 || v < 0 {
			t.Errorf("out of range [0, 10): %v\n", v)
		}
	}
}

func TestExpDecaySample100(t *testing.T) {
	rand.Seed(1)
	s := NewExpDecaySample(1000, 0.01)
	for i := 0; i < 100; i++ {
		s.Update(int64(i))
	}
	if size := s.Count(); 100 != size {
		t.Errorf("s.Count(): 100 != %v\n", size)
	}
	if size := s.Size(); 100 != size {
		t.Errorf("s.Size(): 100 != %v\n", size)
	}
	if l := len(s.Values()); 100 != l {
		t.Errorf("len(s.Values()): 100 != %v\n", l)
	}
	for _, v := range s.Values() {
		if v > 100 || v < 0 {
			t.Errorf("out of range [0, 100): %v\n", v)
		}
	}
}

func TestExpDecaySample1000(t *testing.T) {
	rand.Seed(1)
	s := NewExpDecaySample(100, 0.99)
	for i := 0; i < 1000; i++ {
		s.Update(int64(i))
	}
	if size := s.Count(); 1000 != size {
		t.Errorf("s.Count(): 1000 != %v\n", size)
	}
	if size := s.Size(); 100 != size {
		t.Errorf("s.Size(): 100 != %v\n", size)
	}
	if l := len(s.Values()); 100 != l {
		t.Errorf("len(s.Values()): 100 != %v\n", l)
	}
	for _, v := range s.Values() {
		if v > 1000 || v < 0 {
			t.Errorf("out of range [0, 1000): %v\n", v)
		}
	}
}

// This test makes sure that the sample's priority is not amplified by using
// nanosecond duration since start rather than second duration since start.
// The priority becomes +Inf quickly after starting if this is done,
// effectively freezing the set of samples until a rescale step happens.
func TestExpDecaySampleNanosecondRegression(t *testing.T) {
	rand.Seed(1)
	s := NewExpDecaySample(100, 0.99)
	for i := 0; i < 100; i++ {
		s.Update(10)
	}
	time.Sleep(1 * time.Millisecond)
	for i := 0; i < 100; i++ {
		s.Update(20)
	}
	v := s.Values()
	avg := float64(0)
	for i := 0; i < len(v); i++ {
		avg += float64(v[i])
	}
	avg /= float64(len(v))
	if avg > 16 || avg < 14 {
		t.Errorf("out of range [14, 16]: %v\n", avg)
	}
}

func TestExpDecaySampleRescale(t *testing.T) {
	s := NewExpDecaySample(2, 0.001).(*ExpDecaySample)
	s.update(time.Now(), 1)
	s.update(time.Now().Add(time.Hour+time.Microsecond), 1)
	for _, v := range s.values.Values() {
		if v.k == 0.0 {
			t.Fatal("v.k == 0.0")
		}
	}
}

func TestExpDecaySampleSnapshot(t *testing.T) {
	now := time.Now()
	rand.Seed(1)
	s := NewExpDecaySample(100, 0.99)
	for i := 1; i <= 10000; i++ {
		s.(*ExpDecaySample).update(now.Add(time.Duration(i)), int64(i))
	}
	snapshot := s.Snapshot()
	s.Update(1)
	testExpDecaySampleStatistics(t, snapshot)
}

func TestExpDecaySampleStatistics(t *testing.T) {
	now := time.Now()
	rand.Seed(1)
	s := NewExpDecaySample(100, 0.99)
	for i := 1; i <= 10000; i++ {
		s.(*ExpDecaySample).update(now.Add(time.Duration(i)), int64(i))
	}
	testExpDecaySampleStatistics(t, s)
}

func TestUniformSample(t *testing.T) {
	rand.Seed(1)
	s := NewUniformSample(100)
	for i := 0; i < 1000; i++ {
		s.Update(int64(i))
	}
	if size := s.Count(); 1000 != size {
		t.Errorf("s.Count(): 1000 != %v\n", size)
	}
	if size := s.Size(); 100 != size {
		t.Errorf("s.Size(): 100 != %v\n", size)
	}
	if l := len(s.Values()); 100 != l {
		t.Errorf("len(s.Values()): 100 != %v\n", l)
	}
	for _, v := range s.Values() {
		if v > 1000 || v < 0 {
			t.Errorf("out of range [0, 100): %v\n", v)
		}
	}
}

func TestUniformSampleIncludesTail(t *testing.T) {
	rand.Seed(1)
	s := NewUniformSample(100)
	max := 100
	for i := 0; i < max; i++ {
		s.Update(int64(i))
	}
	v := s.Values()
	sum := 0
	exp := (max - 1) * max / 2
	for i := 0; i < len(v); i++ {
		sum += int(v[i])
	}
	if exp != sum {
		t.Errorf("sum: %v != %v\n", exp, sum)
	}
}

func TestUniformSampleSnapshot(t *testing.T) {
	s := NewUniformSample(100)
	for i := 1; i <= 10000; i++ {
		s.Update(int64(i))
	}
	snapshot := s.Snapshot()
	s.Update(1)
	testUniformSampleStatistics(t, snapshot)
}

func TestUniformSampleStatistics(t *testing.T) {
	rand.Seed(1)
	s := NewUniformSample(100)
	for i := 1; i <= 10000; i++ {
		s.Update(int64(i))
	}
	testUniformSampleStatistics(t, s)
}

func benchmarkSample(b *testing.B, s Sample) {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	pauseTotalNs := memStats.PauseTotalNs
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Update(1)
	}
	b.StopTimer()
	runtime.GC()
	runtime.ReadMemStats(&memStats)
	b.Logf("GC cost: %d ns/op", int(memStats.PauseTotalNs-pauseTotalNs)/b.N)
}

func testExpDecaySampleStatistics(t *testing.T, s Sample) {
	if count := s.Count(); 10000 != count {
		t.Errorf("s.Count(): 10000 != %v\n", count)
	}
	if min := s.Min(); 107 != min {
		t.Errorf("s.Min(): 107 != %v\n", min)
	}
	if max := s.Max(); 10000 != max {
		t.Errorf("s.Max(): 10000 != %v\n", max)
	}
	if mean := s.Mean(); 4965.98 != mean {
		t.Errorf("s.Mean(): 4965.98 != %v\n", mean)
	}
	if stdDev := s.StdDev(); 2959.825156930727 != stdDev {
		t.Errorf("s.StdDev(): 2959.825156930727 != %v\n", stdDev)
	}
	ps := s.Percentiles([]float64{0.5, 0.75, 0.99})
	if 4615 != ps[0] {
		t.Errorf("median: 4615 != %v\n", ps[0])
	}
	if 7672 != ps[1] {
		t.Errorf("75th percentile: 7672 != %v\n", ps[1])
	}
	if 9998.99 != ps[2] {
		t.Errorf("99th percentile: 9998.99 != %v\n", ps[2])
	}
}

func testUniformSampleStatistics(t *testing.T, s Sample) {
	if count := s.Count(); 10000 != count {
		t.Errorf("s.Count(): 10000 != %v\n", count)
	}
	if min := s.Min(); 37 != min {
		t.Errorf("s.Min(): 37 != %v\n", min)
	}
	if max := s.Max(); 9989 != max {
		t.Errorf("s.Max(): 9989 != %v\n", max)
	}
	if mean := s.Mean(); 4748.14 != mean {
		t.Errorf("s.Mean(): 4748.14 != %v\n", mean)
	}
	if stdDev := s.StdDev(); 2826.684117548333 != stdDev {
		t.Errorf("s.StdDev(): 2826.684117548333 != %v\n", stdDev)
	}
	ps := s.Percentiles([]float64{0.5, 0.75, 0.99})
	if 4599 != ps[0] {
		t.Errorf("median: 4599 != %v\n", ps[0])
	}
	if 7380.5 != ps[1] {
		t.Errorf("75th percentile: 7380.5 != %v\n", ps[1])
	}
	if 9986.429999999998 != ps[2] {
		t.Errorf("99th percentile: 9986.429999999998 != %v\n", ps[2])
	}
}

// TestUniformSampleConcurrentUpdateCount would expose data race problems with
// concurrent Update and Count calls on Sample when test is called with -race
// argument
func TestUniformSampleConcurrentUpdateCount(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	s := NewUniformSample(100)
	for i := 0; i < 100; i++ {
		s.Update(int64(i))
	}
	quit := make(chan struct{})
	go func() {
		t := time.NewTicker(10 * time.Millisecond)
		for {
			select {
			case <-t.C:
				s.Update(rand.Int63())
			case <-quit:
				t.Stop()
				return
			}
		}
	}()
	for i := 0; i < 1000; i++ {
		s.Count()
		time.Sleep(5 * time.Millisecond)
	}
	quit <- struct{}{}
}
