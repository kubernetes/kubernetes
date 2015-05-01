/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

const schedulerSubsystem = "scheduler"

var (
	E2eSchedulingLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: schedulerSubsystem,
			Name:      "e2e_scheduling_latency_microseconds",
			Help:      "E2e scheduling latency (scheduling algorith + binding)",
		},
	)
	SchedulingAlgorithmLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: schedulerSubsystem,
			Name:      "scheduling_algorithm_latency_microseconds",
			Help:      "Scheduling algorithm latency",
		},
	)
	BindingLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: schedulerSubsystem,
			Name:      "binding_latency_microseconds",
			Help:      "Binding latency",
		},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(E2eSchedulingLatency)
		prometheus.MustRegister(SchedulingAlgorithmLatency)
		prometheus.MustRegister(BindingLatency)
	})
}

// Gets the time since the specified start in microseconds.
func SinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}
