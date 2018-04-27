/*
Copyright 2015 The Kubernetes Authors.

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

// All the histogram based metrics have 1ms as size for the smallest bucket.
var (
	E2eSchedulingLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: schedulerSubsystem,
			Name:      "e2e_scheduling_latency_microseconds",
			Help:      "E2e scheduling latency (scheduling algorithm + binding)",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	SchedulingAlgorithmLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: schedulerSubsystem,
			Name:      "scheduling_algorithm_latency_microseconds",
			Help:      "Scheduling algorithm latency",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	SchedulingAlgorithmPredicateEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: schedulerSubsystem,
			Name:      "scheduling_algorithm_predicate_evaluation",
			Help:      "Scheduling algorithm predicate evaluation duration",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	SchedulingAlgorithmPriorityEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: schedulerSubsystem,
			Name:      "scheduling_algorithm_priority_evaluation",
			Help:      "Scheduling algorithm priority evaluation duration",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	SchedulingAlgorithmPremptionEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: schedulerSubsystem,
			Name:      "scheduling_algorithm_preemption_evaluation",
			Help:      "Scheduling algorithm preemption evaluation duration",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	BindingLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: schedulerSubsystem,
			Name:      "binding_latency_microseconds",
			Help:      "Binding latency",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	PreemptionVictims = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: schedulerSubsystem,
			Name:      "pod_preemption_victims",
			Help:      "Number of selected preemption victims",
		})
	PreemptionAttempts = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: schedulerSubsystem,
			Name:      "total_preemption_attempts",
			Help:      "Total preemption attempts in the cluster till now",
		})
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(E2eSchedulingLatency)
		prometheus.MustRegister(SchedulingAlgorithmLatency)
		prometheus.MustRegister(BindingLatency)

		prometheus.MustRegister(SchedulingAlgorithmPredicateEvaluationDuration)
		prometheus.MustRegister(SchedulingAlgorithmPriorityEvaluationDuration)
		prometheus.MustRegister(SchedulingAlgorithmPremptionEvaluationDuration)
		prometheus.MustRegister(PreemptionVictims)
		prometheus.MustRegister(PreemptionAttempts)
	})
}

// SinceInMicroseconds gets the time since the specified start in microseconds.
func SinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}
