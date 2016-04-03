package metrics

import (
	"math"
	"testing"
	"time"
)

func TestInmemSink(t *testing.T) {
	inm := NewInmemSink(10*time.Millisecond, 50*time.Millisecond)

	data := inm.Data()
	if len(data) != 1 {
		t.Fatalf("bad: %v", data)
	}

	// Add data points
	inm.SetGauge([]string{"foo", "bar"}, 42)
	inm.EmitKey([]string{"foo", "bar"}, 42)
	inm.IncrCounter([]string{"foo", "bar"}, 20)
	inm.IncrCounter([]string{"foo", "bar"}, 22)
	inm.AddSample([]string{"foo", "bar"}, 20)
	inm.AddSample([]string{"foo", "bar"}, 22)

	data = inm.Data()
	if len(data) != 1 {
		t.Fatalf("bad: %v", data)
	}

	intvM := data[0]
	intvM.RLock()

	if time.Now().Sub(intvM.Interval) > 10*time.Millisecond {
		t.Fatalf("interval too old")
	}
	if intvM.Gauges["foo.bar"] != 42 {
		t.Fatalf("bad val: %v", intvM.Gauges)
	}
	if intvM.Points["foo.bar"][0] != 42 {
		t.Fatalf("bad val: %v", intvM.Points)
	}

	agg := intvM.Counters["foo.bar"]
	if agg.Count != 2 {
		t.Fatalf("bad val: %v", agg)
	}
	if agg.Sum != 42 {
		t.Fatalf("bad val: %v", agg)
	}
	if agg.SumSq != 884 {
		t.Fatalf("bad val: %v", agg)
	}
	if agg.Min != 20 {
		t.Fatalf("bad val: %v", agg)
	}
	if agg.Max != 22 {
		t.Fatalf("bad val: %v", agg)
	}
	if agg.Mean() != 21 {
		t.Fatalf("bad val: %v", agg)
	}
	if agg.Stddev() != math.Sqrt(2) {
		t.Fatalf("bad val: %v", agg)
	}

	if agg.LastUpdated.IsZero() {
		t.Fatalf("agg.LastUpdated is not set: %v", agg)
	}

	diff := time.Now().Sub(agg.LastUpdated).Seconds()
	if diff > 1 {
		t.Fatalf("time diff too great: %f", diff)
	}

	if agg = intvM.Samples["foo.bar"]; agg == nil {
		t.Fatalf("missing sample")
	}

	intvM.RUnlock()

	for i := 1; i < 10; i++ {
		time.Sleep(10 * time.Millisecond)
		inm.SetGauge([]string{"foo", "bar"}, 42)
		data = inm.Data()
		if len(data) != min(i+1, 5) {
			t.Fatalf("bad: %v", data)
		}
	}

	// Should not exceed 5 intervals!
	time.Sleep(10 * time.Millisecond)
	inm.SetGauge([]string{"foo", "bar"}, 42)
	data = inm.Data()
	if len(data) != 5 {
		t.Fatalf("bad: %v", data)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
