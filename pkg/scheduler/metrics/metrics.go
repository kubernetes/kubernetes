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

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	volumeschedulingmetrics "k8s.io/kubernetes/pkg/controller/volume/scheduling/metrics"
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
	scheduleAttempts = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "schedule_attempts_total",
			Help:           "Number of attempts to schedule pods, by the result. 'unschedulable' means a pod could not be scheduled, while 'error' means an internal scheduler problem.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"result"})
	// PodScheduleSuccesses counts how many pods were scheduled.
	PodScheduleSuccesses = scheduleAttempts.With(metrics.Labels{"result": "scheduled"})
	// PodScheduleFailures counts how many pods could not be scheduled.
	PodScheduleFailures = scheduleAttempts.With(metrics.Labels{"result": "unschedulable"})
	// PodScheduleErrors counts how many pods could not be scheduled due to a scheduler error.
	PodScheduleErrors = scheduleAttempts.With(metrics.Labels{"result": "error"})
	SchedulingLatency = metrics.NewSummaryVec(
		&metrics.SummaryOpts{
			Subsystem: SchedulerSubsystem,
			Name:      SchedulingLatencyName,
			Help:      "Scheduling latency in seconds split by sub-parts of the scheduling operation",
			// Make the sliding window of 5h.
			// TODO: The value for this should be based on some SLI definition (long term).
			MaxAge:         5 * time.Hour,
			StabilityLevel: metrics.ALPHA,
		},
		[]string{OperationLabel},
	)
	DeprecatedSchedulingLatency = metrics.NewSummaryVec(
		&metrics.SummaryOpts{
			Subsystem: SchedulerSubsystem,
			Name:      DeprecatedSchedulingLatencyName,
			Help:      "(Deprecated) Scheduling latency in seconds split by sub-parts of the scheduling operation",
			// Make the sliding window of 5h.
			// TODO: The value for this should be based on some SLI definition (long term).
			MaxAge:         5 * time.Hour,
			StabilityLevel: metrics.ALPHA,
		},
		[]string{OperationLabel},
	)
	E2eSchedulingLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "e2e_scheduling_duration_seconds",
			Help:           "E2e scheduling latency in seconds (scheduling algorithm + binding)",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	DeprecatedE2eSchedulingLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "e2e_scheduling_latency_microseconds",
			Help:           "(Deprecated) E2e scheduling latency in microseconds (scheduling algorithm + binding)",
			Buckets:        metrics.ExponentialBuckets(1000, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	SchedulingAlgorithmLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduling_algorithm_duration_seconds",
			Help:           "Scheduling algorithm latency in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	DeprecatedSchedulingAlgorithmLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduling_algorithm_latency_microseconds",
			Help:           "(Deprecated) Scheduling algorithm latency in microseconds",
			Buckets:        metrics.ExponentialBuckets(1000, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	SchedulingAlgorithmPredicateEvaluationDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduling_algorithm_predicate_evaluation_seconds",
			Help:           "Scheduling algorithm predicate evaluation duration in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	DeprecatedSchedulingAlgorithmPredicateEvaluationDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduling_algorithm_predicate_evaluation",
			Help:           "(Deprecated) Scheduling algorithm predicate evaluation duration in microseconds",
			Buckets:        metrics.ExponentialBuckets(1000, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	SchedulingAlgorithmPriorityEvaluationDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduling_algorithm_priority_evaluation_seconds",
			Help:           "Scheduling algorithm priority evaluation duration in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	DeprecatedSchedulingAlgorithmPriorityEvaluationDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduling_algorithm_priority_evaluation",
			Help:           "(Deprecated) Scheduling algorithm priority evaluation duration in microseconds",
			Buckets:        metrics.ExponentialBuckets(1000, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	SchedulingAlgorithmPremptionEvaluationDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduling_algorithm_preemption_evaluation_seconds",
			Help:           "Scheduling algorithm preemption evaluation duration in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	DeprecatedSchedulingAlgorithmPremptionEvaluationDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduling_algorithm_preemption_evaluation",
			Help:           "(Deprecated) Scheduling algorithm preemption evaluation duration in microseconds",
			Buckets:        metrics.ExponentialBuckets(1000, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	BindingLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "binding_duration_seconds",
			Help:           "Binding latency in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	DeprecatedBindingLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "binding_latency_microseconds",
			Help:           "(Deprecated) Binding latency in microseconds",
			Buckets:        metrics.ExponentialBuckets(1000, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	PreemptionVictims = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "pod_preemption_victims",
			Help:      "Number of selected preemption victims",
			// we think #victims>50 is pretty rare, therefore [50, +Inf) is considered a single bucket.
			Buckets:        metrics.LinearBuckets(5, 5, 10),
			StabilityLevel: metrics.ALPHA,
		})
	PreemptionAttempts = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "total_preemption_attempts",
			Help:           "Total preemption attempts in the cluster till now",
			StabilityLevel: metrics.ALPHA,
		})

	pendingPods = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "pending_pods",
			Help:           "Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableQ.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"queue"})

	PodSchedulingDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "pod_scheduling_duration_seconds",
			Help:           "E2e latency for a pod being scheduled which may include multiple scheduling attempts.",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		})

	PodSchedulingAttempts = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "pod_scheduling_attempts",
			Help:           "Number of attempts to successfully schedule a pod.",
			Buckets:        metrics.ExponentialBuckets(1, 2, 5),
			StabilityLevel: metrics.ALPHA,
		})

	FrameworkExtensionPointDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "framework_extension_point_duration_seconds",
			Help:           "Latency for running all plugins of a specific extension point.",
			Buckets:        nil,
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"extension_point", "status"})

	SchedulerQueueIncomingPods = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "queue_incoming_pods_total",
			Help:           "Number of pods added to scheduling queues by event and queue type.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"queue", "event"})

	metricsList = []metrics.Registerable{
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
		PodSchedulingDuration,
		PodSchedulingAttempts,
		FrameworkExtensionPointDuration,
		SchedulerQueueIncomingPods,
	}
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		for _, metric := range metricsList {
			legacyregistry.MustRegister(metric)
		}
		volumeschedulingmetrics.RegisterVolumeSchedulingMetrics()
	})
}

// GetGather returns the gatherer. It used by test case outside current package.
func GetGather() metrics.Gatherer {
	return legacyregistry.DefaultGatherer
}

// ActivePods returns the pending pods metrics with the label active
func ActivePods() metrics.GaugeMetric {
	return pendingPods.With(metrics.Labels{"queue": "active"})
}

// BackoffPods returns the pending pods metrics with the label backoff
func BackoffPods() metrics.GaugeMetric {
	return pendingPods.With(metrics.Labels{"queue": "backoff"})
}

// UnschedulablePods returns the pending pods metrics with the label unschedulable
func UnschedulablePods() metrics.GaugeMetric {
	return pendingPods.With(metrics.Labels{"queue": "unschedulable"})
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
