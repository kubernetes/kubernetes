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
	volumescheduling "k8s.io/kubernetes/pkg/controller/volume/scheduling"
)

const (
	// SchedulerSubsystem - subsystem name used by scheduler
	SchedulerSubsystem = "scheduler"
	// SchedulingLatencyName - scheduler latency metric name
	SchedulingLatencyName = "scheduling_duration_seconds"
	// DeprecatedSchedulingLatencyName - scheduler latency metric name which is deprecated
	DeprecatedSchedulingLatencyName = "scheduling_latency_seconds"

	// OperationLabel - operation label name
	OperationLabel = "operation"
	// Below are possible values for the operation label. Each represents a substep of e2e scheduling:

	// PredicateEvaluation - predicate evaluation operation label value
	PredicateEvaluation = "predicate_evaluation"
	// PriorityEvaluation - priority evaluation operation label value
	PriorityEvaluation = "priority_evaluation"
	// PreemptionEvaluation - preemption evaluation operation label value (occurs in case of scheduling fitError).
	PreemptionEvaluation = "preemption_evaluation"
	// Binding - binding operation label value
	Binding = "binding"
	// E2eScheduling - e2e scheduling operation label value
)

// All the histogram based metrics have 1ms as size for the smallest bucket.
var (
	scheduleAttempts = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "schedule_attempts_total",
			Help:      "Number of attempts to schedule pods, by the result. 'unschedulable' means a pod could not be scheduled, while 'error' means an internal scheduler problem.",
		}, []string{"result"})
	// PodScheduleSuccesses counts how many pods were scheduled.
	PodScheduleSuccesses = scheduleAttempts.With(prometheus.Labels{"result": "scheduled"})
	// PodScheduleFailures counts how many pods could not be scheduled.
	PodScheduleFailures = scheduleAttempts.With(prometheus.Labels{"result": "unschedulable"})
	// PodScheduleErrors counts how many pods could not be scheduled due to a scheduler error.
	PodScheduleErrors = scheduleAttempts.With(prometheus.Labels{"result": "error"})
	SchedulingLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: SchedulerSubsystem,
			Name:      SchedulingLatencyName,
			Help:      "Scheduling latency in seconds split by sub-parts of the scheduling operation",
			// Make the sliding window of 5h.
			// TODO: The value for this should be based on some SLI definition (long term).
			MaxAge: 5 * time.Hour,
		},
		[]string{OperationLabel},
	)
	DeprecatedSchedulingLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: SchedulerSubsystem,
			Name:      DeprecatedSchedulingLatencyName,
			Help:      "(Deprecated) Scheduling latency in seconds split by sub-parts of the scheduling operation",
			// Make the sliding window of 5h.
			// TODO: The value for this should be based on some SLI definition (long term).
			MaxAge: 5 * time.Hour,
		},
		[]string{OperationLabel},
	)
	E2eSchedulingLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "e2e_scheduling_duration_seconds",
			Help:      "E2e scheduling latency in seconds (scheduling algorithm + binding)",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 15),
		},
	)
	DeprecatedE2eSchedulingLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "e2e_scheduling_latency_microseconds",
			Help:      "(Deprecated) E2e scheduling latency in microseconds (scheduling algorithm + binding)",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	SchedulingAlgorithmLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "scheduling_algorithm_duration_seconds",
			Help:      "Scheduling algorithm latency in seconds",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 15),
		},
	)
	DeprecatedSchedulingAlgorithmLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "scheduling_algorithm_latency_microseconds",
			Help:      "(Deprecated) Scheduling algorithm latency in microseconds",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	SchedulingAlgorithmPredicateEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "scheduling_algorithm_predicate_evaluation_seconds",
			Help:      "Scheduling algorithm predicate evaluation duration in seconds",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 15),
		},
	)
	DeprecatedSchedulingAlgorithmPredicateEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "scheduling_algorithm_predicate_evaluation",
			Help:      "(Deprecated) Scheduling algorithm predicate evaluation duration in microseconds",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	SchedulingAlgorithmPriorityEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "scheduling_algorithm_priority_evaluation_seconds",
			Help:      "Scheduling algorithm priority evaluation duration in seconds",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 15),
		},
	)
	DeprecatedSchedulingAlgorithmPriorityEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "scheduling_algorithm_priority_evaluation",
			Help:      "(Deprecated) Scheduling algorithm priority evaluation duration in microseconds",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	SchedulingAlgorithmPremptionEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "scheduling_algorithm_preemption_evaluation_seconds",
			Help:      "Scheduling algorithm preemption evaluation duration in seconds",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 15),
		},
	)
	DeprecatedSchedulingAlgorithmPremptionEvaluationDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "scheduling_algorithm_preemption_evaluation",
			Help:      "(Deprecated) Scheduling algorithm preemption evaluation duration in microseconds",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 15),
		},
	)
	BindingLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "binding_duration_seconds",
			Help:      "Binding latency in seconds",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 15),
		},
	)
	DeprecatedBindingLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "binding_latency_microseconds",
			Help:      "(Deprecated) Binding latency in microseconds",
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

	pendingPods = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "pending_pods",
			Help:      "Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableQ.",
		}, []string{"queue"})
	ActivePods        = pendingPods.With(prometheus.Labels{"queue": "active"})
	BackoffPods       = pendingPods.With(prometheus.Labels{"queue": "backoff"})
	UnschedulablePods = pendingPods.With(prometheus.Labels{"queue": "unschedulable"})

	metricsList = []prometheus.Collector{
		scheduleAttempts,
		SchedulingLatency,
		DeprecatedSchedulingLatency,
		E2eSchedulingLatency,
		DeprecatedE2eSchedulingLatency,
		SchedulingAlgorithmLatency,
		DeprecatedSchedulingAlgorithmLatency,
		BindingLatency,
		DeprecatedBindingLatency,
		SchedulingAlgorithmPredicateEvaluationDuration,
		DeprecatedSchedulingAlgorithmPredicateEvaluationDuration,
		SchedulingAlgorithmPriorityEvaluationDuration,
		DeprecatedSchedulingAlgorithmPriorityEvaluationDuration,
		SchedulingAlgorithmPremptionEvaluationDuration,
		DeprecatedSchedulingAlgorithmPremptionEvaluationDuration,
		PreemptionVictims,
		PreemptionAttempts,
		pendingPods,
	}
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		for _, metric := range metricsList {
			prometheus.MustRegister(metric)
		}

		volumescheduling.RegisterVolumeSchedulingMetrics()
	})
}

// Reset resets metrics
func Reset() {
	SchedulingLatency.Reset()
	DeprecatedSchedulingLatency.Reset()
}

// SinceInMicroseconds gets the time since the specified start in microseconds.
func SinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}
