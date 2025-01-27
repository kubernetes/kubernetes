package health

import (
	compbasemetrics "k8s.io/component-base/metrics"
)

type registerables []compbasemetrics.Registerable

var (
	healthyTargetsTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "health_monitor_healthy_target_total",
			Help:           "Number of healthy instances registered with the health monitor. Partitioned by targets.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"target"},
	)

	currentHealthyTargets = compbasemetrics.NewGauge(
		&compbasemetrics.GaugeOpts{
			Name:           "health_monitor_current_healthy_targets",
			Help:           "Number of currently healthy instances observed by the health monitor",
			StabilityLevel: compbasemetrics.ALPHA,
		},
	)

	unHealthyTargetsTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "health_monitor_unhealthy_target_total",
			Help:           "Number of unhealthy instances registered with the health monitor. Partitioned by targets.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"target"},
	)

	readyzViolationRequestTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "health_monitor_readyz_violation_request_total",
			Help:           "Number of HTTP requests partitioned by status code and target that violate the readyz protocol.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"code", "target"},
	)

	metrics = registerables{
		healthyTargetsTotal,
		currentHealthyTargets,
		unHealthyTargetsTotal,
		readyzViolationRequestTotal,
	}
)

// HealthyTargetsTotal increments the total number of healthy instances observed by the health monitor
func HealthyTargetsTotal(target string) {
	healthyTargetsTotal.WithLabelValues(target).Add(1)
}

// CurrentHealthyTargets keeps track of the current number of healthy targets observed by the health monitor
func CurrentHealthyTargets(count float64) {
	currentHealthyTargets.Set(count)
}

// UnHealthyTargetsTotal increments the total number of unhealthy instances observed by the health monitor
func UnHealthyTargetsTotal(target string) {
	unHealthyTargetsTotal.WithLabelValues(target).Add(1)
}

// ReadyzProtocolRequestTotal increments the total number of requests issues by the health monitor that violate the "readyz" protocol
//
// the "readyz" protocol defines the following HTTP status code:
//
//	HTTP 200 - when the server operates normally
//	HTTP 500 - when the server is not ready, for example, is undergoing a shutdown
func ReadyzProtocolRequestTotal(code, target string) {
	readyzViolationRequestTotal.WithLabelValues(code, target).Add(1)
}

// Metrics specifies a set of methods that are used to register various metrics
type Metrics struct {
	// HealthyTargetsTotal increments the total number of healthy instances observed by the health monitor
	HealthyTargetsTotal func(target string)

	// CurrentHealthyTargets keeps track of the current number of healthy targets observed by the health monitor
	CurrentHealthyTargets func(count float64)

	// UnHealthyTargetsTotal increments the total number of unhealthy instances observed by the health monitor
	UnHealthyTargetsTotal func(target string)

	// ReadyzProtocolRequestTotal increments the total number of requests issues by the health monitor that violate the "readyz" protocol
	//
	// the "readyz" protocol defines the following HTTP status code:
	//   HTTP 200 - when the server operates normally
	//   HTTP 500 - when the server is not ready, for example, is undergoing a shutdown
	ReadyzProtocolRequestTotal func(code, target string)
}

// Register is a way to register the health monitor related metrics in the provided store
func Register(registerFn func(...compbasemetrics.Registerable)) *Metrics {
	registerFn(metrics...)
	return &Metrics{
		HealthyTargetsTotal:        HealthyTargetsTotal,
		CurrentHealthyTargets:      CurrentHealthyTargets,
		UnHealthyTargetsTotal:      UnHealthyTargetsTotal,
		ReadyzProtocolRequestTotal: ReadyzProtocolRequestTotal,
	}
}

type noopMetrics struct{}

func (noopMetrics) TargetsTotal(string)                 {}
func (noopMetrics) TargetsGauge(float64)                {}
func (noopMetrics) TargetsWithCodeTotal(string, string) {}
