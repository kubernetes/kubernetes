package prometheus

import (
	"runtime"
	"runtime/debug"
	"time"
)

type goCollector struct {
	goroutines Gauge
	gcDesc     *Desc
}

// NewGoCollector returns a collector which exports metrics about the current
// go process.
func NewGoCollector() *goCollector {
	return &goCollector{
		goroutines: NewGauge(GaugeOpts{
			Name: "go_goroutines",
			Help: "Number of goroutines that currently exist.",
		}),
		gcDesc: NewDesc(
			"go_gc_duration_seconds",
			"A summary of the GC invocation durations.",
			nil, nil),
	}
}

// Describe returns all descriptions of the collector.
func (c *goCollector) Describe(ch chan<- *Desc) {
	ch <- c.goroutines.Desc()
	ch <- c.gcDesc
}

// Collect returns the current state of all metrics of the collector.
func (c *goCollector) Collect(ch chan<- Metric) {
	c.goroutines.Set(float64(runtime.NumGoroutine()))
	ch <- c.goroutines

	var stats debug.GCStats
	stats.PauseQuantiles = make([]time.Duration, 5)
	debug.ReadGCStats(&stats)

	quantiles := make(map[float64]float64)
	for idx, pq := range stats.PauseQuantiles[1:] {
		quantiles[float64(idx+1)/float64(len(stats.PauseQuantiles)-1)] = pq.Seconds()
	}
	quantiles[0.0] = stats.PauseQuantiles[0].Seconds()
	ch <- MustNewConstSummary(c.gcDesc, uint64(stats.NumGC), float64(stats.PauseTotal.Seconds()), quantiles)
}
