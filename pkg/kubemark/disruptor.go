package kubemark

import (
	"fmt"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/kubernetes/pkg/kubelet"
)

// RuntimeDisrupter allows to report unhealthy state based on the following criteria:
// - periodically switching from Healthy to not Healthy and back (e.g. to simulate a case
//   where Node object switches its Ready condition into false for a period of time to
//   exercise infrastructure activity)
// - switch node Ready condition to false permanently (e.g. to exercise if node controller
//   properly removes and possible drains node before deletion)
type RuntimeDisrupter struct {
	// TurnUnhealthyAfter starts reporting unhealthy after specified period of time
	TurnUnhealthyAfter        bool
	TurnUnhealthyPeriodically bool
	UnhealthyDuration         metav1.Duration
	HealthyDuration           metav1.Duration

	healthy    bool
	healthyErr error
	healthyMux sync.Mutex
	stop       bool
}

// Name for health checker name
func (rd *RuntimeDisrupter) Name() string {
	return "RuntimeDisrupter"
}

// Start starts the runtime disruptor
func (rd *RuntimeDisrupter) Start() {
	rd.healthyMux.Lock()
	rd.healthy = true
	rd.stop = false
	rd.healthyMux.Unlock()

	if rd.TurnUnhealthyAfter {
		// Wait for TurnUnhealthyAfter time, then switch to unhealthy
		go func() {
			time.Sleep(rd.HealthyDuration.Duration)
			rd.healthyMux.Lock()
			defer rd.healthyMux.Unlock()
			rd.healthy = false
			rd.healthyErr = fmt.Errorf("reporting unhealty indefinitely on request (%v after start up)", rd.HealthyDuration.Duration)
		}()
	}
	if rd.TurnUnhealthyPeriodically {
		go func() {
			for {
				time.Sleep(rd.HealthyDuration.Duration)
				rd.healthyMux.Lock()
				rd.healthy = false
				rd.healthyErr = fmt.Errorf("reporting unhealty periodically on request (every %v)", rd.UnhealthyDuration.Duration)
				if rd.stop {
					return
				}
				rd.healthyMux.Unlock()

				time.Sleep(rd.UnhealthyDuration.Duration)
				rd.healthyMux.Lock()
				rd.healthy = true
				if rd.stop {
					return
				}
				rd.healthyMux.Unlock()
			}
		}()
	}
}

// Stop stops the runtime disruptor
func (rd *RuntimeDisrupter) Stop() {
	rd.healthyMux.Lock()
	defer rd.healthyMux.Unlock()
	rd.stop = true
}

// Healthy checks if specific part of runtime is healthy
func (rd *RuntimeDisrupter) Healthy() (bool, error) {
	rd.healthyMux.Lock()
	defer rd.healthyMux.Unlock()
	return rd.healthy, rd.healthyErr
}

var _ kubelet.RuntimeHealthChecker = &RuntimeDisrupter{}
