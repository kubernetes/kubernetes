package route

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	// subsystem is the name of this subsystem used for prometheus metrics.
	subsystem = "route_controller"
)

var registration sync.Once

var (
	routeSyncCount = metrics.NewCounter(&metrics.CounterOpts{
		Name:           "route_sync_total",
		Subsystem:      subsystem,
		Help:           "A metric counting the amount of times routes have been synced with the cloud provider.",
		StabilityLevel: metrics.BETA,
	})
)

func registerMetrics() {
	registration.Do(func() {
		legacyregistry.MustRegister(routeSyncCount)
	})
}
