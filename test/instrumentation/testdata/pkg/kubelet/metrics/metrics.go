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

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
)

// This const block defines the metric names for the kubelet metrics.
const (
	KubeletSubsystem           = "kubelet"
	NodeNameKey                = "node_name"
	NodeLabelKey               = "node"
	PodWorkerDurationKey       = "pod_worker_duration_seconds"
	PodStartDurationKey        = "pod_start_duration_seconds"
	CgroupManagerOperationsKey = "cgroup_manager_duration_seconds"
	PodWorkerStartDurationKey  = "pod_worker_start_duration_seconds"
	PLEGRelistDurationKey      = "pleg_relist_duration_seconds"
	PLEGDiscardEventsKey       = "pleg_discard_events"
	PLEGRelistIntervalKey      = "pleg_relist_interval_seconds"
	PLEGLastSeenKey            = "pleg_last_seen_seconds"
	EvictionsKey               = "evictions"
	EvictionStatsAgeKey        = "eviction_stats_age_seconds"
	PreemptionsKey             = "preemptions"
	RunningPodsKey             = "running_pods"
	RunningContainersKey       = "running_containers"
	// Metrics keys of remote runtime operations
	RuntimeOperationsKey         = "runtime_operations_total"
	RuntimeOperationsDurationKey = "runtime_operations_duration_seconds"
	RuntimeOperationsErrorsKey   = "runtime_operations_errors_total"
	// Metrics keys of device plugin operations
	DevicePluginRegistrationCountKey  = "device_plugin_registration_total"
	DevicePluginAllocationDurationKey = "device_plugin_alloc_duration_seconds"
	// Metrics keys of pod resources operations
	PodResourcesEndpointRequestsTotalKey          = "pod_resources_endpoint_requests_total"
	PodResourcesEndpointRequestsListKey           = "pod_resources_endpoint_requests_list"
	PodResourcesEndpointRequestsGetAllocatableKey = "pod_resources_endpoint_requests_get_allocatable"
	PodResourcesEndpointErrorsListKey             = "pod_resources_endpoint_errors_list"
	PodResourcesEndpointErrorsGetAllocatableKey   = "pod_resources_endpoint_errors_get_allocatable"

	// Metrics keys for RuntimeClass
	RunPodSandboxDurationKey = "run_podsandbox_duration_seconds"
	RunPodSandboxErrorsKey   = "run_podsandbox_errors_total"
)

const (
	// Subsystem names.
	pvControllerSubsystem = "pv_collector"

	// Metric names.
	totalPVKey    = "total_pv_count"
	boundPVKey    = "bound_pv_count"
	unboundPVKey  = "unbound_pv_count"
	boundPVCKey   = "bound_pvc_count"
	unboundPVCKey = "unbound_pvc_count"

	// Label names.
	namespaceLabel    = "namespace"
	storageClassLabel = "storage_class"
	pluginNameLabel   = "plugin_name"
	volumeModeLabel   = "volume_mode"

	// String to use when plugin name cannot be determined
	pluginNameNotAvailable = "N/A"
)

const (
	requestKind         = "request_kind"
	priorityLevel       = "priority_level"
	flowSchema          = "flow_schema"
	phase               = "phase"
	LabelNamePhase      = "phase"
	LabelValueWaiting   = "waiting"
	LabelValueExecuting = "executing"
)

var (
	defObjectives = map[float64]float64{0.5: 0.5, 0.75: 0.75}
	testBuckets   = []float64{0, 0.5, 1.0}
	testLabels    = []string{"a", "b", "c"}
	maxAge        = 2 * time.Minute

	// NodeName is a Gauge that tracks the ode's name. The count is always 1.
	NodeName = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      KubeletSubsystem,
			Name:           NodeNameKey,
			Help:           "The node's name. The count is always 1.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{NodeLabelKey},
	)
	// ContainersPerPodCount is a Histogram that tracks the number of containers per pod.
	ContainersPerPodCount = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           "containers_per_pod_count",
			Help:           "The number of containers per pod.",
			Buckets:        testBuckets,
			StabilityLevel: metrics.ALPHA,
		},
	)
	// PodWorkerDuration is a Histogram that tracks the duration (in seconds) in takes to sync a single pod.
	// Broken down by the operation type.
	PodWorkerDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PodWorkerDurationKey,
			Help:           "Duration in seconds to sync a single pod. Broken down by operation type: create, update, or sync",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
		},
		testLabels,
	)
	// healthcheck is a Prometheus Gauge metrics used for recording the results of a k8s healthcheck.
	healthcheck = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      "kubernetes",
			Name:           "healthcheck",
			Help:           "This metric records the result of a single healthcheck.",
			StabilityLevel: metrics.BETA,
		},
		[]string{"name", "type"},
	)

	// healthchecksTotal is a Prometheus Counter metrics used for counting the results of a k8s healthcheck.
	healthchecksTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      "kubernetes",
			Name:           "healthchecks_total",
			Help:           "This metric records the results of all healthcheck.",
			StabilityLevel: metrics.BETA,
		},
		[]string{"name", "type", "status"},
	)
	// PodWorkerDuration is a Histogram that tracks the duration (in seconds) in takes to sync a single pod.
	// Broken down by the operation type.
	SummaryMaxAge = metrics.NewSummary(
		&metrics.SummaryOpts{
			Subsystem:      KubeletSubsystem,
			Name:           "max_age",
			Help:           "Duration in seconds to sync a single pod. Broken down by operation type: create, update, or sync",
			StabilityLevel: metrics.BETA,
			MaxAge:         2 * time.Hour,
		},
	)

	// PodWorkerDuration is a Histogram that tracks the duration (in seconds) in takes to sync a single pod.
	// Broken down by the operation type.
	SummaryMaxAgeConst = metrics.NewSummary(
		&metrics.SummaryOpts{
			Subsystem:      KubeletSubsystem,
			Name:           "max_age_const",
			Help:           "Duration in seconds to sync a single pod. Broken down by operation type: create, update, or sync",
			StabilityLevel: metrics.BETA,
			MaxAge:         maxAge,
		},
	)
	// PodStartDuration is a Histogram that tracks the duration (in seconds) it takes for a single pod to go from pending to running.
	PodStartDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PodStartDurationKey,
			Help:           "Duration in seconds for a single pod to go from pending to running.",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
		},
	)
	// CgroupManagerDuration is a Histogram that tracks the duration (in seconds) it takes for cgroup manager operations to complete.
	// Broken down by method.
	CgroupManagerDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           CgroupManagerOperationsKey,
			Help:           "Duration in seconds for cgroup manager operations. Broken down by method.",
			Buckets:        metrics.ExponentialBucketsRange(0.01, 10, 10),
			StabilityLevel: metrics.BETA,
		},
		[]string{"operation_type"},
	)
	// PodWorkerStartDuration is a Histogram that tracks the duration (in seconds) it takes from seeing a pod to starting a worker.
	PodWorkerStartDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PodWorkerStartDurationKey,
			Help:           "Duration in seconds from seeing a pod to starting a worker.",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
		},
	)
	// PriorityLevelExecutionSeatsGaugeVec creates observers of seats occupied throughout execution for priority levels
	PriorityLevelExecutionSeatsGaugeVec = metrics.NewTimingHistogramVec(
		&metrics.TimingHistogramOpts{
			Namespace: "namespace",
			Subsystem: "subsystem",
			Name:      "priority_level_seat_utilization",
			Help:      "Observations, at the end of every nanosecond, of utilization of seats for any stage of execution (but only initial stage for WATCHes)",
			// Buckets for both 0.99 and 1.0 mean PromQL's histogram_quantile will reveal saturation
			Buckets:        []float64{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1},
			ConstLabels:    map[string]string{phase: "executing"},
			StabilityLevel: metrics.BETA,
		},
		[]string{"priorityLevel"},
	)

	// PriorityLevelExecutionSeatsGaugeVec creates observers of seats occupied throughout execution for priority levels
	TestConstLabels = metrics.NewTimingHistogramVec(
		&metrics.TimingHistogramOpts{
			Namespace: "test",
			Subsystem: "const",
			Name:      "label",
			Help:      "Observations, at the end of every nanosecond, of utilization of seats for any stage of execution (but only initial stage for WATCHes)",
			// Buckets for both 0.99 and 1.0 mean PromQL's histogram_quantile will reveal saturation
			Buckets:        []float64{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1},
			ConstLabels:    map[string]string{"somestring": "executing", phase: "blah"},
			StabilityLevel: metrics.BETA,
		},
		[]string{"priorityLevel"},
	)
	// PLEGRelistDuration is a Histogram that tracks the duration (in seconds) it takes for relisting pods in the Kubelet's
	// Pod Lifecycle Event Generator (PLEG).
	PLEGRelistDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PLEGRelistDurationKey,
			Help:           "Duration in seconds for relisting pods in PLEG.",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
		},
	)
	// PLEGDiscardEvents is a Counter that tracks the number of discarding events in the Kubelet's Pod Lifecycle Event Generator (PLEG).
	PLEGDiscardEvents = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PLEGDiscardEventsKey,
			Help:           "The number of discard events in PLEG.",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// PLEGRelistInterval is a Histogram that tracks the intervals (in seconds) between relisting in the Kubelet's
	// Pod Lifecycle Event Generator (PLEG).
	PLEGRelistInterval = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           "test_histogram_metric",
			Help:           "Interval in seconds between relisting in PLEG.",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.STABLE,
		},
	)
	// PLEGLastSeen is a Gauge giving the Unix timestamp when the Kubelet's
	// Pod Lifecycle Event Generator (PLEG) was last seen active.
	PLEGLastSeen = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PLEGLastSeenKey,
			Help:           "Timestamp in seconds when PLEG was last seen active.",
			StabilityLevel: metrics.ALPHA,
		},
	)
	// RuntimeOperations is a Counter that tracks the cumulative number of remote runtime operations.
	// Broken down by operation type.
	RuntimeOperations = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           RuntimeOperationsKey,
			Help:           "Cumulative number of runtime operations by operation type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation_type"},
	)
	// RuntimeOperationsDuration is a Histogram that tracks the duration (in seconds) for remote runtime operations to complete.
	// Broken down by operation type.
	RuntimeOperationsDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           RuntimeOperationsDurationKey,
			Help:           "Duration in seconds of runtime operations. Broken down by operation type.",
			Buckets:        metrics.ExponentialBuckets(.005, 2.5, 14),
			StabilityLevel: metrics.BETA,
		},
		[]string{"operation_type"},
	)
	// RuntimeOperationsErrors is a Counter that tracks the cumulative number of remote runtime operations errors.
	// Broken down by operation type.
	RuntimeOperationsErrors = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           RuntimeOperationsErrorsKey,
			Help:           "Cumulative number of runtime operation errors by operation type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation_type"},
	)
	// Evictions is a Counter that tracks the cumulative number of pod evictions initiated by the kubelet.
	// Broken down by eviction signal.
	Evictions = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           EvictionsKey,
			Help:           "Cumulative number of pod evictions by eviction signal",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"eviction_signal"},
	)
	// EvictionStatsAge is a Histogram that tracks the time (in seconds) between when stats are collected and when a pod is evicted
	// based on those stats. Broken down by eviction signal.
	EvictionStatsAge = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           EvictionStatsAgeKey,
			Help:           "Time between when stats are collected, and when pod is evicted based on those stats by eviction signal",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"eviction_signal"},
	)
	// Preemptions is a Counter that tracks the cumulative number of pod preemptions initiated by the kubelet.
	// Broken down by preemption signal. A preemption is only recorded for one resource, the sum of all signals
	// is the number of preemptions on the given node.
	Preemptions = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PreemptionsKey,
			Help:           "Cumulative number of pod preemptions by preemption resource",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"preemption_signal"},
	)
	// MultiLineHelp tests that we can parse multi-line strings
	MultiLineHelp = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      "multiline",
			Help: "Cumulative number of pod preemptions by preemption resource " +
				"asdf asdf asdf " +
				"asdfas dfasdf",
			StabilityLevel: metrics.STABLE,
		},
		[]string{"preemption_signal"},
	)
	// DevicePluginRegistrationCount is a Counter that tracks the cumulative number of device plugin registrations.
	// Broken down by resource name.
	DevicePluginRegistrationCount = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           DevicePluginRegistrationCountKey,
			Help:           "Cumulative number of device plugin registrations. Broken down by resource name.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource_name"},
	)
	// DevicePluginAllocationDuration is a Histogram that tracks the duration (in seconds) to serve a device plugin allocation request.
	// Broken down by resource name.
	DevicePluginAllocationDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           DevicePluginAllocationDurationKey,
			Help:           "Duration in seconds to serve a device plugin Allocation request. Broken down by resource name.",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.BETA,
		},
		[]string{"resource_name"},
	)

	// TestSummary is a Summary that tracks the cumulative number of device plugin registrations.
	// Broken down by resource name.
	TestSummary = metrics.NewSummary(
		&metrics.SummaryOpts{
			Subsystem:      KubeletSubsystem,
			Name:           "summary_metric_test",
			Help:           "Cumulative number of device plugin registrations. Broken down by resource name.",
			StabilityLevel: metrics.STABLE,
		},
	)
	// TestSummaryVec is a NewSummaryVec that tracks the duration (in seconds) to serve a device plugin allocation request.
	// Broken down by resource name.
	TestSummaryVec = metrics.NewSummaryVec(
		&metrics.SummaryOpts{
			Subsystem:      KubeletSubsystem,
			Name:           "summary_vec_metric_test",
			Help:           "Duration in seconds to serve a device plugin Allocation request. Broken down by resource name.",
			Objectives:     defObjectives,
			MaxAge:         metrics.DefMaxAge,
			AgeBuckets:     metrics.DefAgeBuckets,
			BufCap:         metrics.DefBufCap,
			StabilityLevel: metrics.STABLE,
		},
		[]string{"resource_name"},
	)

	// PodResourcesEndpointRequestsTotalCount is a Counter that tracks the cumulative number of requests to the PodResource endpoints.
	// Broken down by server API version.
	PodResourcesEndpointRequestsTotalCount = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PodResourcesEndpointRequestsTotalKey,
			Help:           "Cumulative number of requests to the PodResource endpoint. Broken down by server api version.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"server_api_version"},
	)

	// PodResourcesEndpointRequestsListCount is a Counter that tracks the number of requests to the PodResource List() endpoint.
	// Broken down by server API version.
	PodResourcesEndpointRequestsListCount = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PodResourcesEndpointRequestsListKey,
			Help:           "Number of requests to the PodResource List endpoint. Broken down by server api version.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"server_api_version"},
	)

	// PodResourcesEndpointRequestsGetAllocatableCount is a Counter that tracks the number of requests to the PodResource GetAllocatableResources() endpoint.
	// Broken down by server API version.
	PodResourcesEndpointRequestsGetAllocatableCount = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PodResourcesEndpointRequestsGetAllocatableKey,
			Help:           "Number of requests to the PodResource GetAllocatableResources endpoint. Broken down by server api version.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"server_api_version"},
	)

	// PodResourcesEndpointErrorsListCount is a Counter that tracks the number of errors returned by he PodResource List() endpoint.
	// Broken down by server API version.
	PodResourcesEndpointErrorsListCount = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PodResourcesEndpointErrorsListKey,
			Help:           "Number of requests to the PodResource List endpoint which returned error. Broken down by server api version.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"server_api_version"},
	)

	// PodResourcesEndpointErrorsGetAllocatableCount is a Counter that tracks the number of errors returned by the PodResource GetAllocatableResources() endpoint.
	// Broken down by server API version.
	PodResourcesEndpointErrorsGetAllocatableCount = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           PodResourcesEndpointErrorsGetAllocatableKey,
			Help:           "Number of requests to the PodResource GetAllocatableResources endpoint which returned error. Broken down by server api version.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"server_api_version"},
	)

	// RunPodSandboxDuration is a Histogram that tracks the duration (in seconds) it takes to run Pod Sandbox operations.
	// Broken down by RuntimeClass.Handler.
	RunPodSandboxDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      RunPodSandboxDurationKey,
			Help:      "Duration in seconds of the run_podsandbox operations. Broken down by RuntimeClass.Handler.",
			// Use DefBuckets for now, will customize the buckets if necessary.
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"runtime_handler"},
	)
	// RunPodSandboxErrors is a Counter that tracks the cumulative number of Pod Sandbox operations errors.
	// Broken down by RuntimeClass.Handler.
	RunPodSandboxErrors = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           RunPodSandboxErrorsKey,
			Help:           "Cumulative number of the run_podsandbox operation errors by RuntimeClass.Handler.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"runtime_handler"},
	)

	// RunningPodCount is a gauge that tracks the number of Pods currently with a running sandbox
	// It is used to expose the kubelet internal state: how many pods have running containers in the container runtime, and mainly for debugging purpose.
	RunningPodCount = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      KubeletSubsystem,
			Name:           RunningPodsKey,
			Help:           "Number of pods that have a running pod sandbox",
			StabilityLevel: metrics.ALPHA,
		},
	)
	// RunningContainerCount is a gauge that tracks the number of containers currently running
	RunningContainerCount = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      KubeletSubsystem,
			Name:           RunningContainersKey,
			Help:           "Number of containers currently running",
			StabilityLevel: metrics.BETA,
		},
		[]string{"container_state"},
	)

	NetworkProgrammingLatency = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem: "kube_proxy",
			Name:      "network_programming_duration_seconds",
			Help:      "In Cluster Network Programming Latency in seconds",
			Buckets: merge(
				metrics.LinearBuckets(0.25, 0.25, 2), // 0.25s, 0.50s
				metrics.LinearBuckets(1, 1, 59),      // 1s, 2s, 3s, ... 59s
				metrics.LinearBuckets(60, 5, 12),     // 60s, 65s, 70s, ... 115s
				metrics.LinearBuckets(120, 30, 7),    // 2min, 2.5min, 3min, ..., 5min
			),
			StabilityLevel: metrics.BETA,
		},
	)

	NetworkProgrammingLatency2 = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem: "kube_proxy",
			Name:      "network_programming_duration_seconds2",
			Help:      "In Cluster Network Programming Latency in seconds",
			Buckets: metrics.MergeBuckets(
				metrics.LinearBuckets(0.25, 0.25, 2), // 0.25s, 0.50s
				[]float64{1, 5, 10, 59},              // 1s, 2s, 3s, ... 59s
				metrics.LinearBuckets(60, 5, 12),     // 60s, 65s, 70s, ... 115s
				metrics.LinearBuckets(120, 30, 7),    // 2min, 2.5min, 3min, ..., 5min
			),
			StabilityLevel: metrics.BETA,
		},
	)

	volumeManagerTotalVolumes = "volume_manager_total_volumes"

	_ = metrics.NewDesc(
		volumeManagerTotalVolumes,
		"Number of volumes in Volume Manager",
		[]string{"plugin_name", "state"},
		nil,
		metrics.STABLE, "",
	)

	_ = metrics.NewDesc(
		metrics.BuildFQName("test", "beta", "desc"),
		"Number of volumes in Volume Manager",
		nil,
		map[string]string{"alalala": "lalalal"},
		metrics.BETA, "",
	)
	_ = metrics.NewDesc(
		"test_desc_alpha",
		"Number of volumes in Volume Manager",
		[]string{"plugin_name", "state"},
		map[string]string{"alalala": "lalalal"},
		metrics.ALPHA, "",
	)

	_ = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsCapacityBytesKey),
		"Capacity in bytes of the volume",
		[]string{"namespace", "persistentvolumeclaim"}, nil,
		metrics.BETA, "",
	)
)

var registerMetrics sync.Once

// Register registers all metrics.
func Register(collectors ...metrics.StableCollector) {
	// Register the metrics.
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(NodeName)
		legacyregistry.MustRegister(PodWorkerDuration)
		legacyregistry.MustRegister(PodStartDuration)
		legacyregistry.MustRegister(healthcheck)
		legacyregistry.MustRegister(healthchecksTotal)
		legacyregistry.MustRegister(CgroupManagerDuration)
		legacyregistry.MustRegister(PodWorkerStartDuration)
		legacyregistry.MustRegister(ContainersPerPodCount)
		legacyregistry.MustRegister(PLEGRelistDuration)
		legacyregistry.MustRegister(PLEGDiscardEvents)
		legacyregistry.MustRegister(PLEGRelistInterval)
		legacyregistry.MustRegister(PLEGLastSeen)
		legacyregistry.MustRegister(RuntimeOperations)
		legacyregistry.MustRegister(RuntimeOperationsDuration)
		legacyregistry.MustRegister(RuntimeOperationsErrors)
		legacyregistry.MustRegister(Evictions)
		legacyregistry.MustRegister(EvictionStatsAge)
		legacyregistry.MustRegister(Preemptions)
		legacyregistry.MustRegister(DevicePluginRegistrationCount)
		legacyregistry.MustRegister(DevicePluginAllocationDuration)
		legacyregistry.MustRegister(RunningContainerCount)
		legacyregistry.MustRegister(RunningPodCount)
		legacyregistry.MustRegister(RunPodSandboxDuration)
		legacyregistry.MustRegister(RunPodSandboxErrors)
		legacyregistry.MustRegister(NetworkProgrammingLatency)
		for _, collector := range collectors {
			legacyregistry.CustomMustRegister(collector)
		}
		legacyregistry.RawMustRegister(metrics.NewGaugeFunc(
			&metrics.GaugeOpts{
				Subsystem: "kubelet",
				Name:      "certificate_manager_client_ttl_seconds",
				Help: "Gauge of the TTL (time-to-live) of the Kubelet's client certificate. " +
					"The value is in seconds until certificate expiry (negative if already expired). " +
					"If client certificate is invalid or unused, the value will be +INF.",
				StabilityLevel: metrics.BETA,
			},
			func() float64 {
				return 0
			},
		))
		_ = metrics.Labels{
			"probe_type": "1",
			"container":  "2",
			"pod":        "podName",
			"namespace":  "space",
			"pod_uid":    "123",
		}
	})
}

// GetGather returns the gatherer. It used by test case outside current package.
func GetGather() metrics.Gatherer {
	return legacyregistry.DefaultGatherer
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}

// SetNodeName sets the NodeName Gauge to 1.
func SetNodeName(name types.NodeName) {
	NodeName.WithLabelValues(string(name)).Set(1)
}

func Blah() metrics.ObserverMetric {
	return EvictionStatsAge.With(metrics.Labels{"plugins": "ASDf"})
}

func merge(slices ...[]float64) []float64 {
	result := make([]float64, 1)
	for _, s := range slices {
		result = append(result, s...)
	}
	return result
}
