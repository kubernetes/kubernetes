package metrics

import (
	"runtime"
	"time"
)

func (m *Metrics) SetGauge(key []string, val float32) {
	if m.HostName != "" && m.EnableHostname {
		key = insert(0, m.HostName, key)
	}
	if m.EnableTypePrefix {
		key = insert(0, "gauge", key)
	}
	if m.ServiceName != "" {
		key = insert(0, m.ServiceName, key)
	}
	m.sink.SetGauge(key, val)
}

func (m *Metrics) EmitKey(key []string, val float32) {
	if m.EnableTypePrefix {
		key = insert(0, "kv", key)
	}
	if m.ServiceName != "" {
		key = insert(0, m.ServiceName, key)
	}
	m.sink.EmitKey(key, val)
}

func (m *Metrics) IncrCounter(key []string, val float32) {
	if m.EnableTypePrefix {
		key = insert(0, "counter", key)
	}
	if m.ServiceName != "" {
		key = insert(0, m.ServiceName, key)
	}
	m.sink.IncrCounter(key, val)
}

func (m *Metrics) AddSample(key []string, val float32) {
	if m.EnableTypePrefix {
		key = insert(0, "sample", key)
	}
	if m.ServiceName != "" {
		key = insert(0, m.ServiceName, key)
	}
	m.sink.AddSample(key, val)
}

func (m *Metrics) MeasureSince(key []string, start time.Time) {
	if m.EnableTypePrefix {
		key = insert(0, "timer", key)
	}
	if m.ServiceName != "" {
		key = insert(0, m.ServiceName, key)
	}
	now := time.Now()
	elapsed := now.Sub(start)
	msec := float32(elapsed.Nanoseconds()) / float32(m.TimerGranularity)
	m.sink.AddSample(key, msec)
}

// Periodically collects runtime stats to publish
func (m *Metrics) collectStats() {
	for {
		time.Sleep(m.ProfileInterval)
		m.emitRuntimeStats()
	}
}

// Emits various runtime statsitics
func (m *Metrics) emitRuntimeStats() {
	// Export number of Goroutines
	numRoutines := runtime.NumGoroutine()
	m.SetGauge([]string{"runtime", "num_goroutines"}, float32(numRoutines))

	// Export memory stats
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	m.SetGauge([]string{"runtime", "alloc_bytes"}, float32(stats.Alloc))
	m.SetGauge([]string{"runtime", "sys_bytes"}, float32(stats.Sys))
	m.SetGauge([]string{"runtime", "malloc_count"}, float32(stats.Mallocs))
	m.SetGauge([]string{"runtime", "free_count"}, float32(stats.Frees))
	m.SetGauge([]string{"runtime", "heap_objects"}, float32(stats.HeapObjects))
	m.SetGauge([]string{"runtime", "total_gc_pause_ns"}, float32(stats.PauseTotalNs))
	m.SetGauge([]string{"runtime", "total_gc_runs"}, float32(stats.NumGC))

	// Export info about the last few GC runs
	num := stats.NumGC

	// Handle wrap around
	if num < m.lastNumGC {
		m.lastNumGC = 0
	}

	// Ensure we don't scan more than 256
	if num-m.lastNumGC >= 256 {
		m.lastNumGC = num - 255
	}

	for i := m.lastNumGC; i < num; i++ {
		pause := stats.PauseNs[i%256]
		m.AddSample([]string{"runtime", "gc_pause_ns"}, float32(pause))
	}
	m.lastNumGC = num
}

// Inserts a string value at an index into the slice
func insert(i int, v string, s []string) []string {
	s = append(s, "")
	copy(s[i+1:], s[i:])
	s[i] = v
	return s
}
