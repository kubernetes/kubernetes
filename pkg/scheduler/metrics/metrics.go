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

const (
	// SchedulerSubsystem - subsystem name used by scheduler
	SchedulerSubsystem = "scheduler"
	// SchedulingLatencyName - scheduler latency metric name
	SchedulingLatencyName = "scheduling_latencies_summary"

	// OperationLabel - operation label name
	OperationLabel = "operation"
	// Binding - binding operation label value
	Binding = "binding"
	// SchedulingAlgorithm - scheduling algorithm operation label value
	SchedulingAlgorithm = "scheduling_algorithm"
	// E2eScheduling - e2e scheduling operation label value
	E2eScheduling = "e2e_scheduling"
)

// All the histogram based metrics have 1ms as size for the smallest bucket.
var (
	SchedulingLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: SchedulerSubsystem,
			Name:      SchedulingLatencyName,
			Help:      "Scheduling latency in microseconds split by sub-parts of the scheduling operation",
			// Make the sliding window of 5h.
			// TODO: The value for this should be based on some SLI definition (long term).
			MaxAge: 5 * time.Hour,
		},
		[]string{OperationLabel},
	)
	SchedulingAlgorithmPredicateEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "scheduling_algorithm_predicate_evaluation",
			Help:      "Scheduling algorithm predicate evaluation duration",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	SchedulingAlgorithmPriorityEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "scheduling_algorithm_priority_evaluation",
			Help:      "Scheduling algorithm priority evaluation duration",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	SchedulingAlgorithmPremptionEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "scheduling_algorithm_preemption_evaluation",
			Help:      "Scheduling algorithm preemption evaluation duration",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	PreemptionVictims = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "pod_preemption_victims",
			Help:      "Number of selected preemption victims",
		})
	PreemptionAttempts = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "total_preemption_attempts",
			Help:      "Total preemption attempts in the cluster till now",
		})
	metricsList = []prometheus.Collector{
		SchedulingLatency,
		SchedulingAlgorithmPredicateEvaluationDuration,
		SchedulingAlgorithmPriorityEvaluationDuration,
		SchedulingAlgorithmPremptionEvaluationDuration,
		PreemptionVictims,
		PreemptionAttempts,
	}
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	registerMetrics.Do(func() {
		for _, metric := range metricsList {
			prometheus.MustRegister(metric)
		}
	})
}

// Reset resets metrics
func Reset() {
	SchedulingLatency.Reset()
}

// SinceInMicroseconds gets the time since the specified start in microseconds.
func SinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}
