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
	resourceclaimmetrics "k8s.io/dynamic-resource-allocation/resourceclaim/metrics"
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
	PlacementGenerate,
	PlacementFeasible,
	PodGroupPostFilter,
}

const (
	PreFilter                        = "PreFilter"
	Filter                           = "Filter"
	PreFilterExtensionAddPod         = "PreFilterExtensionAddPod"
	PreFilterExtensionRemovePod      = "PreFilterExtensionRemovePod"
	PostFilter                       = "PostFilter"
	PreScore                         = "PreScore"
	Score                            = "Score"
	ScoreExtensionNormalize          = "ScoreExtensionNormalize"
	PreBind                          = "PreBind"
	PreBindPreFlight                 = "PreBindPreFlight"
	Bind                             = "Bind"
	PostBind                         = "PostBind"
	Reserve                          = "Reserve"
	Unreserve                        = "Unreserve"
	Permit                           = "Permit"
	Sign                             = "Sign"
	PlacementGenerate                = "PlacementGenerate"
	PlacementFeasible                = "PlacementFeasible"
	PlacementScore                   = "PlacementScore"
	PlacementScoreExtensionNormalize = "PlacementScoreExtensionNormalize"
	PodGroupPostFilter               = "PodGroupPostFilter"
)

const (
	QueueingHintResultQueue     = "Queue"
	QueueingHintResultQueueSkip = "QueueSkip"
	QueueingHintResultError     = "Error"
)

// Entity label values used for queued_entities and queue_incoming_entities metrics.
const (
	Pod      = "pod"
	PodGroup = "podgroup"
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
	BatchFlushPodNominated    = "pod_nominated"
	BatchFlushNodeMissing     = "node_missing"
	BatchFlushEmptyList       = "empty_list"
	BatchFlushExpired         = "expired"
	BatchFlushPodIncompatible = "pod_incompatible"
	BatchFlushPodNotBatchable = "pod_not_batchable"
	BatchFlushPreScoreError   = "prescore_error"
	BatchFlushRescoreError    = "rescore_error"
	BatchFlushNormalizeError  = "normalize_error"
)

// DRADeviceBindingConditions status labels
const (
	BindingConditionsStatusSuccess = "success"
	BindingConditionsStatusFailed  = "failure"
	BindingConditionsStatusTimeout = "timeout"
	BindingConditionsStatusError   = "error"
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
	QueuedEntities               *metrics.GaugeVec
	InFlightEvents               *metrics.GaugeVec
	Goroutines                   *metrics.GaugeVec
	BatchAttemptStats            *metrics.CounterVec
	BatchCacheFlushed            *metrics.CounterVec
	BatchRescoreAttempts         *metrics.CounterVec
	BatchRescoreDuration         *metrics.HistogramVec
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

	queueingHintExecutionDuration  *metrics.HistogramVec
	SchedulerQueueIncomingPods     *metrics.CounterVec
	SchedulerQueueIncomingEntities *metrics.CounterVec

	// The below two are only available when the async-preemption feature gate is enabled.
	PreemptionGoroutinesDuration       *metrics.HistogramVec
	PreemptionGoroutinesExecutionTotal *metrics.CounterVec

	// The below are only available when the SchedulerAsyncAPICalls feature gate is enabled.
	AsyncAPICallsTotal   *metrics.CounterVec
	AsyncAPICallDuration *metrics.HistogramVec
	AsyncAPIPendingCalls *metrics.GaugeVec

	// The below is only available when the DRAExtendedResource feature gate is enabled.
	// This is the same metric that also gets recorded in the kube-controller-manager.
	ResourceClaimCreatesTotal = resourceclaimmetrics.ResourceClaimCreate

	podGroupScheduleAttempts           *metrics.CounterVec
	podGroupSchedulingLatency          *metrics.HistogramVec
	PodGroupSchedulingAlgorithmLatency *metrics.Histogram
	// The below are only available when the DRADeviceBindingConditions feature gate is enabled.
	DRABindingConditionsAllocationsTotal *metrics.CounterVec
	DRABindingConditionsPreBindDuration  *metrics.HistogramVec

	WorkloadPreemptionAttempts    *metrics.CounterVec
	WorkloadPreemptionVictims     *metrics.Histogram
	PreemptionWorkloadDisruptions *metrics.HistogramVec
	PreemptionEvaluationDuration  *metrics.HistogramVec
	PreemptionExecutionDuration   *metrics.HistogramVec
	PreemptionPDBViolations       *metrics.CounterVec

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
			resourceclaimmetrics.RegisterMetrics()
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.GenericWorkload) {
			RegisterMetrics(
				podGroupScheduleAttempts,
				podGroupSchedulingLatency,
				PodGroupSchedulingAlgorithmLatency,
				WorkloadPreemptionAttempts,
				WorkloadPreemptionVictims,
				PreemptionWorkloadDisruptions,
				PreemptionEvaluationDuration,
				PreemptionExecutionDuration,
				PreemptionPDBViolations,
			)
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceBindingConditions) {
			RegisterMetrics(
				DRABindingConditionsAllocationsTotal,
				DRABindingConditionsPreBindDuration,
			)
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
			StabilityLevel: metrics.BETA,
		},
	)
	PreemptionVictims = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "preemption_victims",
			Help:      "Number of selected preemption victims for preemption initiated by a single pod",
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
			Help:           "Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated; 'incomplete' means number of pods in incompletePodGroupPods; 'pending' means number of pods in pendingPodGroupPods.",
			StabilityLevel: metrics.STABLE,
		}, []string{"queue"})
	QueuedEntities = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "queued_entities",
			Help:           "Number of queued scheduling entities ('pod' or 'podgroup'; 'pod' stands for individual pods that are not members of any podgroup) by the queue type. 'active' means number of entities in activeQ; 'backoff' means number of entities in backoffQ; 'unschedulable' means number of entities in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable entities that the scheduler never attempted to schedule because they are gated.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"queue", "type"})
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
			StabilityLevel: metrics.BETA,
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
	BatchRescoreAttempts = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "batch_rescore_attempts_total",
			Help:           "Counts of rescore attempts during opportunistic batching.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"profile"})

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
			Help:           "Number of pods that were successfully scheduled after being flushed from unschedulableEntities due to timeout. This metric helps detect potential queueing hint misconfigurations or event handling issues.",
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
			StabilityLevel: metrics.BETA,
		},
		[]string{"plugin", "extension_point", "status"})

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

	SchedulerQueueIncomingEntities = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "queue_incoming_entities_total",
			Help:           "Number of scheduling entities added to scheduling queues by event, queue type, and entity type. Entity types are either 'pod' (for individual pods that are not members of any podgroup) or 'podgroup'.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"queue", "event", "type"})

	PermitWaitDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "permit_wait_duration_seconds",
			Help:           "Duration of waiting on permit.",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.BETA,
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
			StabilityLevel: metrics.BETA,
		}, []string{"plugin", "profile"})

	PluginEvaluationTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "plugin_evaluation_total",
			Help:           "Number of attempts to schedule pods by each plugin and the extension point (available only in PreFilter, Filter, PreScore, and Score).",
			StabilityLevel: metrics.BETA,
		}, []string{"plugin", "extension_point", "profile"})

	PreemptionGoroutinesDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:         SchedulerSubsystem,
			Name:              "preemption_goroutines_duration_seconds",
			Help:              "Duration in seconds for running goroutines for the preemption.",
			Buckets:           metrics.ExponentialBuckets(0.01, 2, 20),
			StabilityLevel:    metrics.ALPHA,
			DeprecatedVersion: "1.37.0",
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

	DRABindingConditionsAllocationsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "dra_bindingconditions_allocations_total",
			Help:           "Number of allocations using devices with BindingConditions, counted per driver per scheduling attempt",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"profile", "driver", "status"},
	)

	DRABindingConditionsPreBindDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "dra_bindingconditions_wait_duration_seconds",
			Help:           "Time in seconds spent waiting for BindingConditions to be satisfied during PreBind.",
			Buckets:        metrics.ExponentialBuckets(0.1, 2, 14),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"profile", "driver", "status"},
	)

	GetNodeHintDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "get_node_hint_duration_seconds",
			Help:      "Latency for getting a node hint.",
			// Start with 0.01ms with the last bucket being [~20ms, Inf)
			Buckets:        metrics.ExponentialBuckets(0.00001, 2, 12),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"hinted", "profile"})

	StoreScheduleResultsDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "store_schedule_results_duration_seconds",
			Help:      "Latency for storing scheduling results.",
			// Start with 0.01ms with the last bucket being [~20ms, Inf)
			Buckets:        metrics.ExponentialBuckets(0.00001, 2, 12),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"profile"})

	BatchRescoreDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "batch_rescore_duration_seconds",
			Help:      "Latency for rescoring a node during opportunistic batching.",
			// Start with 0.01ms with the last bucket being [~20ms, Inf)
			Buckets:        metrics.ExponentialBuckets(0.00001, 2, 12),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"profile"})

	// The below (podGroupScheduleAttempts, podGroupSchedulingLatency and PodGroupSchedulingAlgorithmLatency) are only available when the GenericWorkload feature gate is enabled.
	podGroupScheduleAttempts = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "podgroup_schedule_attempts_total",
			Help:           "Number of attempts to schedule pod group, by the result. 'unschedulable' means a pod group could not be scheduled, while 'error' means an internal scheduler problem.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"result", "profile"})
	podGroupSchedulingLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "podgroup_scheduling_attempt_duration_seconds",
			Help:           "Pod group scheduling attempt latency in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		}, []string{"result", "profile"})
	PodGroupSchedulingAlgorithmLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "podgroup_scheduling_algorithm_duration_seconds",
			Help:           "Pod group scheduling algorithm latency in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		})

	// Workload preemption
	WorkloadPreemptionAttempts = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "workload_preemption_attempts_total",
			Help:           "Total preemption attempts initiated by workload (including pod groups) in the cluster till now.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"result"})
	WorkloadPreemptionVictims = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "workload_preemption_victims",
			Help:      "Number of pod preemption victims caused by workload preemption.",
			// Start with 1 with the last bucket being [1024, Inf)
			Buckets:        metrics.ExponentialBuckets(1, 2, 11),
			StabilityLevel: metrics.ALPHA,
		})
	PreemptionWorkloadDisruptions = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "preemption_workload_disruptions",
			Help:      "Number of workload preemption units being preempted. A single preemption unit can be all pods in a pod group (in case of DisruptionMode=all), or a single pod (in case of DisruptionMode=single).",
			// Start with 1 with the last bucket being [1024, Inf)
			Buckets:        metrics.ExponentialBuckets(1, 2, 11),
			StabilityLevel: metrics.ALPHA,
		}, []string{"preemptor"})
	PreemptionEvaluationDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "preemption_evaluation_duration_seconds",
			Help:      "Duration in seconds for identifying the target preemption victims.",
			// Start with 1ms with the last bucket being [~32.8s, Inf)
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 16),
			StabilityLevel: metrics.ALPHA,
		}, []string{"preemptor", "result"})
	PreemptionExecutionDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: SchedulerSubsystem,
			Name:      "preemption_execution_duration_seconds",
			Help:      "Duration in seconds for preempting the target preemption victims. With async preemption enabled, preemption execution does not block the scheduling of other pods.",
			// Start with 1ms with the last bucket being [~32.8s, Inf)
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 16),
			StabilityLevel: metrics.ALPHA,
		}, []string{"preemptor", "result"})
	PreemptionPDBViolations = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      SchedulerSubsystem,
			Name:           "preemption_pdb_violations_total",
			Help:           "Total number of pod disruption budget violations caused by preemption.",
			StabilityLevel: metrics.ALPHA,
		}, []string{"preemptor"},
	)

	metricsList = []metrics.Registerable{
		scheduleAttempts,
		schedulingLatency,
		SchedulingAlgorithmLatency,
		EventHandlingLatency,
		PreemptionVictims,
		PreemptionAttempts,
		pendingPods,
		QueuedEntities,
		PodSchedulingSLIDuration,
		PodSchedulingAttempts,
		PodScheduledAfterFlush,
		FrameworkExtensionPointDuration,
		PluginExecutionDuration,
		SchedulerQueueIncomingPods,
		SchedulerQueueIncomingEntities,
		Goroutines,
		PermitWaitDuration,
		CacheSize,
		unschedulableReasons,
		PluginEvaluationTotal,
		BatchAttemptStats,
		BatchCacheFlushed,
		BatchRescoreAttempts,
		BatchRescoreDuration,
		GetNodeHintDuration,
		StoreScheduleResultsDuration,
		queueingHintExecutionDuration,
		InFlightEvents,
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

// IncompletePodGroupPods returns the pending pods metric with the queue label set to "incomplete".
func IncompletePodGroupPods() metrics.GaugeMetric {
	return pendingPods.With(metrics.Labels{"queue": "incomplete"})
}

// PendingPodGroupPods returns the pending pods metric with the queue label set to "pending".
func PendingPodGroupPods() metrics.GaugeMetric {
	return pendingPods.With(metrics.Labels{"queue": "pending"})
}

// ActiveEntities returns the queued entities metric with the queue label set to "active" and type label set to "Pod" or "PodGroup".
func ActiveEntities(entityType string) metrics.GaugeMetric {
	return QueuedEntities.With(metrics.Labels{"queue": "active", "type": entityType})
}

// BackoffEntities returns the queued entities metric with the queue label set to "backoff" and type label set to "Pod" or "PodGroup".
func BackoffEntities(entityType string) metrics.GaugeMetric {
	return QueuedEntities.With(metrics.Labels{"queue": "backoff", "type": entityType})
}

// UnschedulableEntities returns the queued entities metric with the queue label set to "unschedulable" and type label set to "Pod" or "PodGroup".
func UnschedulableEntities(entityType string) metrics.GaugeMetric {
	return QueuedEntities.With(metrics.Labels{"queue": "unschedulable", "type": entityType})
}

// GatedEntities returns the queued entities metric with the queue label set to "gated" and type label set to "Pod" or "PodGroup".
func GatedEntities(entityType string) metrics.GaugeMetric {
	return QueuedEntities.With(metrics.Labels{"queue": "gated", "type": entityType})
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}

func UnschedulableReason(plugin string, profile string) metrics.GaugeMetric {
	return unschedulableReasons.With(metrics.Labels{"plugin": plugin, "profile": profile})
}
