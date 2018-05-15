package fake

import (
	"github.com/prometheus/client_golang/prometheus"
)

// DummyResettableCollector is a struct holding a dummy resettable
// metric and group implementation.
type DummyResettableCollector struct {
	name        string
	resetCalled int
}

// Describe is a dummy implementation of Prometheus' Collector interface.
func (c *DummyResettableCollector) Describe(d chan<- *prometheus.Desc) {
	d <- prometheus.NewDesc(c.name, "dummy", nil, nil)
}

// Collect is a dummy implementation of Prometheus' Collector interface.
func (c *DummyResettableCollector) Collect(chan<- prometheus.Metric) {}

// Reset increments an internal counter on every call.
func (c *DummyResettableCollector) Reset() {
	c.resetCalled++
}

// ResetCalledCount returns the number of Reset() calls.
func (c *DummyResettableCollector) ResetCalledCount() int {
	return c.resetCalled
}

// New creates a new instance of ResettableCollector.
func New(name string) *DummyResettableCollector {
	return &DummyResettableCollector{
		name: name,
	}
}
