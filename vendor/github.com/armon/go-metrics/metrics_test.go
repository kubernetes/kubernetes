package metrics

import (
	"reflect"
	"runtime"
	"testing"
	"time"
)

func mockMetric() (*MockSink, *Metrics) {
	m := &MockSink{}
	met := &Metrics{sink: m}
	return m, met
}

func TestMetrics_SetGauge(t *testing.T) {
	m, met := mockMetric()
	met.SetGauge([]string{"key"}, float32(1))
	if m.keys[0][0] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}

	m, met = mockMetric()
	met.HostName = "test"
	met.EnableHostname = true
	met.SetGauge([]string{"key"}, float32(1))
	if m.keys[0][0] != "test" || m.keys[0][1] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}

	m, met = mockMetric()
	met.EnableTypePrefix = true
	met.SetGauge([]string{"key"}, float32(1))
	if m.keys[0][0] != "gauge" || m.keys[0][1] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}

	m, met = mockMetric()
	met.ServiceName = "service"
	met.SetGauge([]string{"key"}, float32(1))
	if m.keys[0][0] != "service" || m.keys[0][1] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}
}

func TestMetrics_EmitKey(t *testing.T) {
	m, met := mockMetric()
	met.EmitKey([]string{"key"}, float32(1))
	if m.keys[0][0] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}

	m, met = mockMetric()
	met.EnableTypePrefix = true
	met.EmitKey([]string{"key"}, float32(1))
	if m.keys[0][0] != "kv" || m.keys[0][1] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}

	m, met = mockMetric()
	met.ServiceName = "service"
	met.EmitKey([]string{"key"}, float32(1))
	if m.keys[0][0] != "service" || m.keys[0][1] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}
}

func TestMetrics_IncrCounter(t *testing.T) {
	m, met := mockMetric()
	met.IncrCounter([]string{"key"}, float32(1))
	if m.keys[0][0] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}

	m, met = mockMetric()
	met.EnableTypePrefix = true
	met.IncrCounter([]string{"key"}, float32(1))
	if m.keys[0][0] != "counter" || m.keys[0][1] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}

	m, met = mockMetric()
	met.ServiceName = "service"
	met.IncrCounter([]string{"key"}, float32(1))
	if m.keys[0][0] != "service" || m.keys[0][1] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}
}

func TestMetrics_AddSample(t *testing.T) {
	m, met := mockMetric()
	met.AddSample([]string{"key"}, float32(1))
	if m.keys[0][0] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}

	m, met = mockMetric()
	met.EnableTypePrefix = true
	met.AddSample([]string{"key"}, float32(1))
	if m.keys[0][0] != "sample" || m.keys[0][1] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}

	m, met = mockMetric()
	met.ServiceName = "service"
	met.AddSample([]string{"key"}, float32(1))
	if m.keys[0][0] != "service" || m.keys[0][1] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] != 1 {
		t.Fatalf("")
	}
}

func TestMetrics_MeasureSince(t *testing.T) {
	m, met := mockMetric()
	met.TimerGranularity = time.Millisecond
	n := time.Now()
	met.MeasureSince([]string{"key"}, n)
	if m.keys[0][0] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] > 0.1 {
		t.Fatalf("")
	}

	m, met = mockMetric()
	met.TimerGranularity = time.Millisecond
	met.EnableTypePrefix = true
	met.MeasureSince([]string{"key"}, n)
	if m.keys[0][0] != "timer" || m.keys[0][1] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] > 0.1 {
		t.Fatalf("")
	}

	m, met = mockMetric()
	met.TimerGranularity = time.Millisecond
	met.ServiceName = "service"
	met.MeasureSince([]string{"key"}, n)
	if m.keys[0][0] != "service" || m.keys[0][1] != "key" {
		t.Fatalf("")
	}
	if m.vals[0] > 0.1 {
		t.Fatalf("")
	}
}

func TestMetrics_EmitRuntimeStats(t *testing.T) {
	runtime.GC()
	m, met := mockMetric()
	met.emitRuntimeStats()

	if m.keys[0][0] != "runtime" || m.keys[0][1] != "num_goroutines" {
		t.Fatalf("bad key %v", m.keys)
	}
	if m.vals[0] <= 1 {
		t.Fatalf("bad val: %v", m.vals)
	}

	if m.keys[1][0] != "runtime" || m.keys[1][1] != "alloc_bytes" {
		t.Fatalf("bad key %v", m.keys)
	}
	if m.vals[1] <= 40000 {
		t.Fatalf("bad val: %v", m.vals)
	}

	if m.keys[2][0] != "runtime" || m.keys[2][1] != "sys_bytes" {
		t.Fatalf("bad key %v", m.keys)
	}
	if m.vals[2] <= 100000 {
		t.Fatalf("bad val: %v", m.vals)
	}

	if m.keys[3][0] != "runtime" || m.keys[3][1] != "malloc_count" {
		t.Fatalf("bad key %v", m.keys)
	}
	if m.vals[3] <= 100 {
		t.Fatalf("bad val: %v", m.vals)
	}

	if m.keys[4][0] != "runtime" || m.keys[4][1] != "free_count" {
		t.Fatalf("bad key %v", m.keys)
	}
	if m.vals[4] <= 100 {
		t.Fatalf("bad val: %v", m.vals)
	}

	if m.keys[5][0] != "runtime" || m.keys[5][1] != "heap_objects" {
		t.Fatalf("bad key %v", m.keys)
	}
	if m.vals[5] <= 100 {
		t.Fatalf("bad val: %v", m.vals)
	}

	if m.keys[6][0] != "runtime" || m.keys[6][1] != "total_gc_pause_ns" {
		t.Fatalf("bad key %v", m.keys)
	}
	if m.vals[6] <= 100000 {
		t.Fatalf("bad val: %v", m.vals)
	}

	if m.keys[7][0] != "runtime" || m.keys[7][1] != "total_gc_runs" {
		t.Fatalf("bad key %v", m.keys)
	}
	if m.vals[7] < 1 {
		t.Fatalf("bad val: %v", m.vals)
	}

	if m.keys[8][0] != "runtime" || m.keys[8][1] != "gc_pause_ns" {
		t.Fatalf("bad key %v", m.keys)
	}
	if m.vals[8] <= 1000 {
		t.Fatalf("bad val: %v", m.vals)
	}
}

func TestInsert(t *testing.T) {
	k := []string{"hi", "bob"}
	exp := []string{"hi", "there", "bob"}
	out := insert(1, "there", k)
	if !reflect.DeepEqual(exp, out) {
		t.Fatalf("bad insert %v %v", exp, out)
	}
}
