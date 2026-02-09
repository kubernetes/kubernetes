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

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/kubernetes/pkg/features"
	volumebindingmetrics "k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding/metrics"
)

const (
	// SchedulerSubsystem - subsystem name used by scheduler.
	SchedulerSubsystem = "scheduler"
)

// Below are possible values for the work and operation label.
const (
	// PrioritizingExtender - prioritizing extender work/operation label value.
	PrioritizingExtender = "prioritizing_extender"
	// Binding - binding work/operation label value.
	Binding = "binding"
)

const (
	GoroutineResultSuccess = "success"
	GoroutineResultError   = "error"
)

// ExtensionPoints is a list of possible values for the extension_point label.
var ExtensionPoints = []string{
	PreFilter,
	Filter,
	PreFilterExtensionAddPod,
	PreFilterExtensionRemovePod,
	PostFilter,
	PreScore,
	Score,
	ScoreExtensionNormalize,
	PreBind,
	PreBindPreFlight,
	Bind,
	PostBind,
	Reserve,
	Unreserve,
	Permit,
	Sign,
}

const (
	PreFilter                   = "PreFilter"
	Filter                      = "Filter"
	PreFilterExtensionAddPod    = "PreFilterExtensionAddPod"
	PreFilterExtensionRemovePod = "PreFilterExtensionRemovePod"
	PostFilter                  = "PostFilter"
	PreScore                    = "PreScore"
	Score                       = "Score"
	ScoreExtensionNormalize     = "ScoreExtensionNormalize"
	PreBind                     = "PreBind"
	PreBindPreFlight            = "PreBindPreFlight"
	Bind                        = "Bind"
	PostBind                    = "PostBind"
	Reserve                     = "Reserve"
	Unreserve                   = "Unreserve"
	Permit                      = "Permit"
	Sign                        = "Sign"
)

const (
	QueueingHintResultQueue     = "Queue"
	QueueingHintResultQueueSkip = "QueueSkip"
	QueueingHintResultError     = "Error"
)

const (
	PodPoppedInFlightEvent = "PodPopped"
)

// Possible batch attempt results
const (
	BatchAttemptNoHint      = "no_hint"
	BatchAttemptHintUsed    = "hint_used"
	BatchAttemptHintNotUsed = "hint_not_used"
)

// Possible batch cache flush reasons
const (
	BatchFlushPodFailed       = "pod_failed"
	BatchFlushPodSkipped      = "pod_skipped"
	BatchFlushNodeMissing     = "node_missing"
	BatchFlushNodeNotFull     = "node_not_full"
	BatchFlushEmptyList       = "empty_list"
	BatchFlushExpired         = "expired"
	BatchFlushPodIncompatible = "pod_incompatible"
	BatchFlushPodNotBatchable = "pod_not_batchable"
)

// All the histogram based metrics have 1ms as size for the smallest bucket.
var (
	scheduleAttempts             *metrics.CounterVec
	EventHandlingLatency         *metrics.HistogramVec
	schedulingLatency            *metrics.HistogramVec
	SchedulingAlgorithmLatency   *metrics.Histogram
	PreemptionVictims            *metrics.Histogram
	PreemptionAttempts           *metrics.Counter
	pendingPods                  *metrics.GaugeVec
	InFlightEvents               *metrics.GaugeVec
	Goroutines                   *metrics.GaugeVec
	BatchAttemptStats            *metrics.CounterVec
	BatchCacheFlushed            *metrics.CounterVec
	GetNodeHintDuration          *metrics.HistogramVec
	StoreScheduleResultsDuration *metrics.HistogramVec

	PodSchedulingSLIDuration        *metrics.HistogramVec
	PodSchedulingAttempts           *metrics.Histogram
	PodScheduledAfterFlush          *metrics.Counter
	FrameworkExtensionPointDuration *metrics.HistogramVec
	PluginExecutionDuration         *metrics.HistogramVec

	PermitWaitDuration    *metrics.HistogramVec
	CacheSize             *metrics.GaugeVec
	unschedulableReasons  *metrics.GaugeVec
	PluginEvaluationTotal *metrics.CounterVec

	// The below two are only available when the QHint feature gate is enabled.
	queueingHintExecutionDuration *metrics.HistogramVec
	SchedulerQueueIncomingPods    *metrics.CounterVec

	// The below two are only available when the async-preemption feature gate is enabled.
	PreemptionGoroutinesDuration       *metrics.HistogramVec
	PreemptionGoroutinesExecutionTotal *metrics.CounterVec

	// The below are only available when the SchedulerAsyncAPICalls feature gate is enabled.
	AsyncAPICallsTotal   *metrics.CounterVec
	AsyncAPICallDuration *metrics.HistogramVec
	AsyncAPIPendingCalls *metrics.GaugeVec

	// The below is only available when the DRAExtendedResource feature gate is enabled.
	ResourceClaimCreatesTotal *metrics.CounterVec

	// metricsList is a list of all metrics that should be registered always, regardless of any feature gate's value.
	metricsList []metrics.Registerable
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		InitMetrics()
		RegisterMetrics(metricsList...)
		volumebindingmetrics.RegisterVolumeSchedulingMetrics()

		if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerQueueingHints) {
			RegisterMetrics(queueingHintExecutionDuration, InFlightEvents)
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerAsyncPreemption) {
			RegisterMetrics(PreemptionGoroutinesDuration, PreemptionGoroutinesExecutionTotal)
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerAsyncAPICalls) {
			RegisterMetrics(
				AsyncAPICallsTotal,
				AsyncAPICallDuration,
				AsyncAPIPendingCalls,
			)
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource) {
			RegisterMetrics(ResourceClaimCreatesTotal)
		}
	})
}

func InitMetrics() {
	scheduleAttempts = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "schedule_attempts_total",
			Help:           "Number of attempts to schedule pods, by the result. 'unschedulable' means a pod could not be scheduled, while 'error' means an internal scheduler problem.",
			StabilityLevel: metrics.STABLE,
		}, []string{"result", "profile"})

	EventHandlingLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "event_handling_duration_seconds",
			Help:      "Event handling latency in seconds.",
			// Start with 0.1ms with the last bucket being [~200ms, Inf)
			Buckets:        metrics.ExponentialBuckets(0.0001, 2, 12),
			StabilityLevel: metrics.ALPHA,
		}, []string{"event"})

	schedulingLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "scheduling_attempt_duration_seconds",
			Help:           "Scheduling attempt latency in seconds (scheduling algorithm + binding)",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.STABLE,
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
	PreemptionVictims = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "preemption_victims",
			Help:      "Number of selected preemption victims",
			// we think #victims>64 is pretty rare, therefore [64, +Inf) is considered a single bucket.
			Buckets:        metrics.ExponentialBuckets(1, 2, 7),
			StabilityLevel: metrics.STABLE,
		})
	PreemptionAttempts = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "preemption_attempts_total",
			Help:           "Total preemption attempts in the cluster till now",
			StabilityLevel: metrics.STABLE,
		})
	pendingPods = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "pending_pods",
			Help:           "Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulablePods that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.",
			StabilityLevel: metrics.STABLE,
		}, []string{"queue"})
	InFlightEvents = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "inflight_events",
			Help:           "Number of events currently tracked in the scheduling queue.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"event"})
	Goroutines = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "goroutines",
			Help:           "Number of running goroutines split by the work they do such as binding.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"operation"})
	BatchAttemptStats = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "batch_attempts_total",
			Help:           "Counts of results when we attempt to use batching.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"profile", "result"})
	BatchCacheFlushed = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "batch_cache_flushed_total",
			Help:           "Counts of cache flushes by reason.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"profile", "reason"})

	PodSchedulingSLIDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "pod_scheduling_sli_duration_seconds",
			Help:      "E2e latency for a pod being scheduled, from the time the pod enters the scheduling queue and might involve multiple scheduling attempts.",
			// Start with 10ms with the last bucket being [~88m, Inf).
			Buckets:        metrics.ExponentialBuckets(0.01, 2, 20),
			StabilityLevel: metrics.BETA,
		},
		[]string{"attempts"})

	PodSchedulingAttempts = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "pod_scheduling_attempts",
			Help:           "Number of attempts to successfully schedule a pod.",
			Buckets:        metrics.ExponentialBuckets(1, 2, 5),
			StabilityLevel: metrics.STABLE,
		})

	PodScheduledAfterFlush = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "pod_scheduled_after_flush_total",
			Help:           "Number of pods that were successfully scheduled after being flushed from unschedulablePods due to timeout. This metric helps detect potential queueing hint misconfigurations or event handling issues.",
			StabilityLevel: metrics.ALPHA,
		})

	FrameworkExtensionPointDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "framework_extension_point_duration_seconds",
			Help:      "Latency for running all plugins of a specific extension point.",
			// Start with 0.1ms with the last bucket being [~200ms, Inf)
			Buckets:        metrics.ExponentialBuckets(0.0001, 2, 12),
			StabilityLevel: metrics.STABLE,
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

	// This is only available when the QHint feature gate is enabled.
	queueingHintExecutionDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "queueing_hint_execution_duration_seconds",
			Help:      "Duration for running a queueing hint function of a plugin.",
			// Start with 0.01ms with the last bucket being [~22ms, Inf). We use a small factor (1.5)
			// so that we have better granularity since plugin latency is very sensitive.
			Buckets:        metrics.ExponentialBuckets(0.00001, 1.5, 20),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"plugin", "event", "hint"})

	SchedulerQueueIncomingPods = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "queue_incoming_pods_total",
			Help:           "Number of pods added to scheduling queues by event and queue type.",
			StabilityLevel: metrics.STABLE,
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
			Name:           "cache_size",
			Help:           "Number of nodes, pods, and assumed (bound) pods in the scheduler cache.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"type"})

	unschedulableReasons = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "unschedulable_pods",
			Help:           "The number of unschedulable pods broken down by plugin name. A pod will increment the gauge for all plugins that caused it to not schedule and so this metric have meaning only when broken down by plugin.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"plugin", "profile"})

	PluginEvaluationTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "plugin_evaluation_total",
			Help:           "Number of attempts to schedule pods by each plugin and the extension point (available only in PreFilter, Filter, PreScore, and Score).",
			StabilityLevel: metrics.ALPHA,
		}, []string{"plugin", "extension_point", "profile"})

	PreemptionGoroutinesDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "preemption_goroutines_duration_seconds",
			Help:           "Duration in seconds for running goroutines for the preemption.",
			Buckets:        metrics.ExponentialBuckets(0.01, 2, 20),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result"})

	PreemptionGoroutinesExecutionTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "preemption_goroutines_execution_total",
			Help:           "Number of preemption goroutines executed.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result"})

	// The below (AsyncAPICallsTotal, AsyncAPICallDuration and AsyncAPIPendingCalls) are only available when the SchedulerAsyncAPICalls feature gate is enabled.
	AsyncAPICallsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "async_api_call_execution_total",
			Help:           "Total number of API calls executed by the async dispatcher.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"call_type", "result"})

	AsyncAPICallDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "async_api_call_execution_duration_seconds",
			Help:           "Duration in seconds for executing API call in the async dispatcher.",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"call_type", "result"})

	AsyncAPIPendingCalls = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "pending_async_api_calls",
			Help:           "Number of API calls currently pending in the async queue.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"call_type"})

	ResourceClaimCreatesTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "resourceclaim_creates_total",
			Help:           "Number of ResourceClaims creation requests within scheduler",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status"})

	GetNodeHintDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "get_node_hint_duration_seconds",
			Help:      "Latency for getting a node hint.",
			// Start with 0.01ms with the last bucket being [~200ms, Inf)
			Buckets:        metrics.ExponentialBuckets(0.00001, 2, 12),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"hinted", "profile"})

	StoreScheduleResultsDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "store_schedule_results_duration_seconds",
			Help:      "Latency for getting a no.",
			// Start with 0.01ms with the last bucket being [~200ms, Inf)
			Buckets:        metrics.ExponentialBuckets(0.00001, 2, 12),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"profile"})

	metricsList = []metrics.Registerable{
		scheduleAttempts,
		schedulingLatency,
		SchedulingAlgorithmLatency,
		EventHandlingLatency,
		PreemptionVictims,
		PreemptionAttempts,
		pendingPods,
		PodSchedulingSLIDuration,
		PodSchedulingAttempts,
		PodScheduledAfterFlush,
		FrameworkExtensionPointDuration,
		PluginExecutionDuration,
		SchedulerQueueIncomingPods,
		Goroutines,
		PermitWaitDuration,
		CacheSize,
		unschedulableReasons,
		PluginEvaluationTotal,
		BatchAttemptStats,
		BatchCacheFlushed,
		GetNodeHintDuration,
		StoreScheduleResultsDuration,
	}
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

// GatedPods returns the pending pods metrics with the label gated
func GatedPods() metrics.GaugeMetric {
	return pendingPods.With(metrics.Labels{"queue": "gated"})
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}

func UnschedulableReason(plugin string, profile string) metrics.GaugeMetric {
	return unschedulableReasons.With(metrics.Labels{"plugin": plugin, "profile": profile})
}
