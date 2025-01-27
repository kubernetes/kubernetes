package health

import "time"

// WithUnHealthyProbesThreshold specifies consecutive failed health checks after which a target is considered unhealthy
func (sm *Prober) WithUnHealthyProbesThreshold(unhealthyProbesThreshold int) *Prober {
	sm.unhealthyProbesThreshold = unhealthyProbesThreshold
	return sm
}

// WithHealthyProbesThreshold  specifies consecutive successful health checks after which a target is considered healthy
func (sm *Prober) WithHealthyProbesThreshold(healthyProbesThreshold int) *Prober {
	sm.healthyProbesThreshold = healthyProbesThreshold
	return sm
}

// WithProbeResponseTimeout specifies a time limit for requests made by the HTTP client for the health check
func (sm *Prober) WithProbeResponseTimeout(probeResponseTimeout time.Duration) *Prober {
	sm.client.Timeout = probeResponseTimeout
	return sm
}

// WithProbeInterval specifies a time interval at which health checks are send
func (sm *Prober) WithProbeInterval(probeInterval time.Duration) *Prober {
	sm.probeInterval = probeInterval
	return sm
}

// WithMetrics specifies a set of methods that are used to register various metrics
func (sm *Prober) WithMetrics(metrics *Metrics) *Prober {
	sm.metrics = metrics
	return sm
}
