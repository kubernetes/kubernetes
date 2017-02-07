/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package daemon

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

const (
	daemonSetUpdateSubsystem = "daemonset_rollingupdate"
	podKillCreateLatencyKey  = "pod_kill_create_latency_microseconds"
	podKillRunningLatencyKey = "pod_kill_running_latency_microseconds"
)

var (
	podKillCreateLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: daemonSetUpdateSubsystem,
			Name:      podKillCreateLatencyKey,
			Help:      "Time in milliseconds of a killed daemon pod on a node being created during daemonset rolling update",
		},
	)
	podKillRunningLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: daemonSetUpdateSubsystem,
			Name:      podKillRunningLatencyKey,
			Help:      "Time in milliseconds of a killed daemon pod on a node being created and running during daemonset rolling update",
		},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(podKillCreateLatency)
		prometheus.MustRegister(podKillRunningLatency)
	})
}

func subInMilliseconds(start time.Time, end time.Time) float64 {
	return float64(start.Sub(end).Nanoseconds() / time.Millisecond.Nanoseconds())
}
