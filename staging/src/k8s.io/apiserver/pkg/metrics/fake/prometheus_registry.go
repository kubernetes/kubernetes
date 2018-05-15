package fake

import (
	"github.com/prometheus/client_golang/prometheus"
)

// FakeRegistry is a struct that implements the Prometheus'
// Registry interterface.
type FakeRegistry struct {
	registredMetrics []prometheus.Collector
}

// Register adds a Prometheus metric to an internal registry.
// It does not fail.
func (r *FakeRegistry) Register(collector prometheus.Collector) error {
	r.registredMetrics = append(r.registredMetrics, collector)
	return nil
}

// MustRegister adds Prometheus metrics to an internal registry.
func (r *FakeRegistry) MustRegister(collector ...prometheus.Collector) {
	r.registredMetrics = append(r.registredMetrics, collector...)
}

// Unregister does nothing and always returns true.
func (r *FakeRegistry) Unregister(collector prometheus.Collector) bool {
	return true
}

// GetRegistredMetrics returns a list of all registred metrics.
func (r *FakeRegistry) GetRegistredMetrics() []prometheus.Collector {
	return r.registredMetrics
}
