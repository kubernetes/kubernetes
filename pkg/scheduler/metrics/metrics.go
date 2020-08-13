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
	// DeprecatedSchedulingDurationName - scheduler duration metric name which is deprecated
	DeprecatedSchedulingDurationName = "scheduling_duration_seconds"

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
		}, []string{"result", "profile"})
	DeprecatedSchedulingDuration = metrics.NewSummaryVec(
		&metrics.SummaryOpts{
			Subsystem: SchedulerSubsystem,
			Name:      DeprecatedSchedulingDurationName,
			Help:      "Scheduling latency in seconds split by sub-parts of the scheduling operation (Deprecated since 1.19.0)",
			// Make the sliding window of 5h.
			// TODO: The value for this should be based on some SLI definition (long term).
			MaxAge:            5 * time.Hour,
			StabilityLevel:    metrics.ALPHA,
			DeprecatedVersion: "1.19.0",
		},
		[]string{OperationLabel},
	)
	e2eSchedulingLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "e2e_scheduling_duration_seconds",
			Help:           "E2e scheduling latency in seconds (scheduling algorithm + binding)",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		}, []string{"result", "profile"})
	SchedulingAlgorithmLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduling_algorithm_duration_seconds",
			Help:           "Scheduling algorithm latency in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)
	DeprecatedSchedulingAlgorithmPredicateEvaluationSecondsDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:         SchedulerSubsystem,
			Name:              "scheduling_algorithm_predicate_evaluation_seconds",
			Help:              "Scheduling algorithm predicate evaluation duration in seconds (Deprecated since 1.19.0)",
			Buckets:           metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel:    metrics.ALPHA,
			DeprecatedVersion: "1.19.0",
		},
	)
	DeprecatedSchedulingAlgorithmPriorityEvaluationSecondsDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:         SchedulerSubsystem,
			Name:              "scheduling_algorithm_priority_evaluation_seconds",
			Help:              "Scheduling algorithm priority evaluation duration in seconds (Deprecated since 1.19.0)",
			Buckets:           metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel:    metrics.ALPHA,
			DeprecatedVersion: "1.19.0",
		},
	)
	SchedulingAlgorithmPreemptionEvaluationDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduling_algorithm_preemption_evaluation_seconds",
			Help:           "Scheduling algorithm preemption evaluation duration in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
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
	PreemptionVictims = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "preemption_victims",
			Help:      "Number of selected preemption victims",
			// we think #victims>50 is pretty rare, therefore [50, +Inf) is considered a single bucket.
			Buckets:        metrics.LinearBuckets(5, 5, 10),
			StabilityLevel: metrics.ALPHA,
		})
	PreemptionAttempts = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "preemption_attempts_total",
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
	SchedulerGoroutines = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduler_goroutines",
			Help:           "Number of running goroutines split by the work they do such as binding.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"work"})

	PodSchedulingDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "pod_scheduling_duration_seconds",
			Help:      "E2e latency for a pod being scheduled which may include multiple scheduling attempts.",
			// Start with 10ms with the last bucket being [~88m, Inf).
			Buckets:        metrics.ExponentialBuckets(0.01, 2, 20),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"attempts"})

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
			Subsystem: SchedulerSubsystem,
			Name:      "framework_extension_point_duration_seconds",
			Help:      "Latency for running all plugins of a specific extension point.",
			// Start with 0.1ms with the last bucket being [~200ms, Inf)
			Buckets:        metrics.ExponentialBuckets(0.0001, 2, 12),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"extension_point", "status", "profile"})

	PluginExecutionDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "plugin_execution_duration_seconds",
			Help:      "Duration for running a plugin at a specific extension point.",
			// Start with 0.01ms with the last bucket being [~22ms, Inf). We use a small factor (1.5)
			// so that we have better granularity since plugin latency is very sensitive.
			Buckets:        metrics.ExponentialBuckets(0.00001, 1.5, 20),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"plugin", "extension_point", "status"})

	SchedulerQueueIncomingPods = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "queue_incoming_pods_total",
			Help:           "Number of pods added to scheduling queues by event and queue type.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"queue", "event"})

	PermitWaitDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "permit_wait_duration_seconds",
			Help:           "Duration of waiting on permit.",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result"})

	CacheSize = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduler_cache_size",
			Help:           "Number of nodes, pods, and assumed (bound) pods in the scheduler cache.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"type"})

	metricsList = []metrics.Registerable{
		scheduleAttempts,
		DeprecatedSchedulingDuration,
		e2eSchedulingLatency,
		SchedulingAlgorithmLatency,
		BindingLatency,
		DeprecatedSchedulingAlgorithmPredicateEvaluationSecondsDuration,
		DeprecatedSchedulingAlgorithmPriorityEvaluationSecondsDuration,
		SchedulingAlgorithmPreemptionEvaluationDuration,
		PreemptionVictims,
		PreemptionAttempts,
		pendingPods,
		PodSchedulingDuration,
		PodSchedulingAttempts,
		FrameworkExtensionPointDuration,
		PluginExecutionDuration,
		SchedulerQueueIncomingPods,
		SchedulerGoroutines,
		PermitWaitDuration,
		CacheSize,
	}
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		RegisterMetrics(metricsList...)
		volumeschedulingmetrics.RegisterVolumeSchedulingMetrics()
	})
}

// RegisterMetrics registers a list of metrics.
// This function is exported because it is intended to be used by out-of-tree plugins to register their custom metrics.
func RegisterMetrics(extraMetrics ...metrics.Registerable) {
	for _, metric := range extraMetrics {
		legacyregistry.MustRegister(metric)
	}
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
	DeprecatedSchedulingDuration.Reset()
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}
