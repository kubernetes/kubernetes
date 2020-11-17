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
	"fmt"
	"sync"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
)

// This const block defines the metric names for the kubelet metrics.
const (
	KubeletSubsystem             = "kubelet"
	NodeNameKey                  = "node_name"
	NodeLabelKey                 = "node"
	PodWorkerDurationKey         = "pod_worker_duration_seconds"
	PodStartDurationKey          = "pod_start_duration_seconds"
	CgroupManagerOperationsKey   = "cgroup_manager_duration_seconds"
	PodWorkerStartDurationKey    = "pod_worker_start_duration_seconds"
	PLEGRelistDurationKey        = "pleg_relist_duration_seconds"
	PLEGDiscardEventsKey         = "pleg_discard_events"
	PLEGRelistIntervalKey        = "pleg_relist_interval_seconds"
	PLEGLastSeenKey              = "pleg_last_seen_seconds"
	EvictionsKey                 = "evictions"
	EvictionStatsAgeKey          = "eviction_stats_age_seconds"
	PreemptionsKey               = "preemptions"
	VolumeStatsCapacityBytesKey  = "volume_stats_capacity_bytes"
	VolumeStatsAvailableBytesKey = "volume_stats_available_bytes"
	VolumeStatsUsedBytesKey      = "volume_stats_used_bytes"
	VolumeStatsInodesKey         = "volume_stats_inodes"
	VolumeStatsInodesFreeKey     = "volume_stats_inodes_free"
	VolumeStatsInodesUsedKey     = "volume_stats_inodes_used"
	// Metrics keys of remote runtime operations
	RuntimeOperationsKey         = "runtime_operations_total"
	RuntimeOperationsDurationKey = "runtime_operations_duration_seconds"
	RuntimeOperationsErrorsKey   = "runtime_operations_errors_total"
	// Metrics keys of device plugin operations
	DevicePluginRegistrationCountKey  = "device_plugin_registration_total"
	DevicePluginAllocationDurationKey = "device_plugin_alloc_duration_seconds"
	// Metrics keys of pod resources operations
	PodResourcesEndpointRequestsTotalKey = "pod_resources_endpoint_requests_total"

	// Metric keys for node config
	AssignedConfigKey             = "node_config_assigned"
	ActiveConfigKey               = "node_config_active"
	LastKnownGoodConfigKey        = "node_config_last_known_good"
	ConfigErrorKey                = "node_config_error"
	ConfigSourceLabelKey          = "node_config_source"
	ConfigSourceLabelValueLocal   = "local"
	ConfigUIDLabelKey             = "node_config_uid"
	ConfigResourceVersionLabelKey = "node_config_resource_version"
	KubeletConfigKeyLabelKey      = "node_config_kubelet_key"

	// Metrics keys for RuntimeClass
	RunPodSandboxDurationKey = "run_podsandbox_duration_seconds"
	RunPodSandboxErrorsKey   = "run_podsandbox_errors_total"
)

var (
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
	// ContainersPerPodCount is a Counter that tracks the number of containers per pod.
	ContainersPerPodCount = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           "containers_per_pod_count",
			Help:           "The number of containers per pod.",
			Buckets:        metrics.DefBuckets,
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
		[]string{"operation_type"},
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
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
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
			Name:           PLEGRelistIntervalKey,
			Help:           "Interval in seconds between relisting in PLEG.",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
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
			StabilityLevel: metrics.ALPHA,
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
			StabilityLevel: metrics.ALPHA,
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

	// Metrics for node config

	// AssignedConfig is a Gauge that is set 1 if the Kubelet has a NodeConfig assigned.
	AssignedConfig = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      KubeletSubsystem,
			Name:           AssignedConfigKey,
			Help:           "The node's understanding of intended config. The count is always 1.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{ConfigSourceLabelKey, ConfigUIDLabelKey, ConfigResourceVersionLabelKey, KubeletConfigKeyLabelKey},
	)
	// ActiveConfig is a Gauge that is set to 1 if the Kubelet has an active NodeConfig.
	ActiveConfig = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      KubeletSubsystem,
			Name:           ActiveConfigKey,
			Help:           "The config source the node is actively using. The count is always 1.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{ConfigSourceLabelKey, ConfigUIDLabelKey, ConfigResourceVersionLabelKey, KubeletConfigKeyLabelKey},
	)
	// LastKnownGoodConfig is a Gauge that is set to 1 if the Kubelet has a NodeConfig it can fall back to if there
	// are certain errors.
	LastKnownGoodConfig = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      KubeletSubsystem,
			Name:           LastKnownGoodConfigKey,
			Help:           "The config source the node will fall back to when it encounters certain errors. The count is always 1.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{ConfigSourceLabelKey, ConfigUIDLabelKey, ConfigResourceVersionLabelKey, KubeletConfigKeyLabelKey},
	)
	// ConfigError is a Gauge that is set to 1 if the node is experiencing a configuration-related error.
	ConfigError = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      KubeletSubsystem,
			Name:           ConfigErrorKey,
			Help:           "This metric is true (1) if the node is experiencing a configuration-related error, false (0) otherwise.",
			StabilityLevel: metrics.ALPHA,
		},
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

	// RunningPodCount is a gauge that tracks the number of Pods currently running
	RunningPodCount = metrics.NewGauge(
		&metrics.GaugeOpts{
			Subsystem:      KubeletSubsystem,
			Name:           "running_pods",
			Help:           "Number of pods currently running",
			StabilityLevel: metrics.ALPHA,
		},
	)
	// RunningContainerCount is a gauge that tracks the number of containers currently running
	RunningContainerCount = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      KubeletSubsystem,
			Name:           "running_containers",
			Help:           "Number of containers currently running",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"container_state"},
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
		if utilfeature.DefaultFeatureGate.Enabled(features.DynamicKubeletConfig) {
			legacyregistry.MustRegister(AssignedConfig)
			legacyregistry.MustRegister(ActiveConfig)
			legacyregistry.MustRegister(LastKnownGoodConfig)
			legacyregistry.MustRegister(ConfigError)
		}
		for _, collector := range collectors {
			legacyregistry.CustomMustRegister(collector)
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

const configMapAPIPathFmt = "/api/v1/namespaces/%s/configmaps/%s"

func configLabels(source *corev1.NodeConfigSource) (map[string]string, error) {
	if source == nil {
		return map[string]string{
			// prometheus requires all of the labels that can be set on the metric
			ConfigSourceLabelKey:          "local",
			ConfigUIDLabelKey:             "",
			ConfigResourceVersionLabelKey: "",
			KubeletConfigKeyLabelKey:      "",
		}, nil
	}
	if source.ConfigMap != nil {
		return map[string]string{
			ConfigSourceLabelKey:          fmt.Sprintf(configMapAPIPathFmt, source.ConfigMap.Namespace, source.ConfigMap.Name),
			ConfigUIDLabelKey:             string(source.ConfigMap.UID),
			ConfigResourceVersionLabelKey: source.ConfigMap.ResourceVersion,
			KubeletConfigKeyLabelKey:      source.ConfigMap.KubeletConfigKey,
		}, nil
	}
	return nil, fmt.Errorf("unrecognized config source type, all source subfields were nil")
}

// track labels across metric updates, so we can delete old label sets and prevent leaks
var assignedConfigLabels map[string]string

// SetAssignedConfig tracks labels according to the assigned NodeConfig. It also tracks labels
// across metric updates so old labels can be safely deleted.
func SetAssignedConfig(source *corev1.NodeConfigSource) error {
	// compute the timeseries labels from the source
	labels, err := configLabels(source)
	if err != nil {
		return err
	}
	// clean up the old timeseries (WithLabelValues creates a new one for each distinct label set)
	if !AssignedConfig.Delete(assignedConfigLabels) {
		klog.Warningf("Failed to delete metric for labels %v. This may result in ambiguity from multiple metrics concurrently indicating different assigned configs.", assignedConfigLabels)
	}
	// record the new timeseries
	assignedConfigLabels = labels
	// expose the new timeseries with a constant count of 1
	AssignedConfig.With(assignedConfigLabels).Set(1)
	return nil
}

// track labels across metric updates, so we can delete old label sets and prevent leaks
var activeConfigLabels map[string]string

// SetActiveConfig tracks labels according to the NodeConfig that is currently used by the Kubelet.
// It also tracks labels across metric updates so old labels can be safely deleted.
func SetActiveConfig(source *corev1.NodeConfigSource) error {
	// compute the timeseries labels from the source
	labels, err := configLabels(source)
	if err != nil {
		return err
	}
	// clean up the old timeseries (WithLabelValues creates a new one for each distinct label set)
	if !ActiveConfig.Delete(activeConfigLabels) {
		klog.Warningf("Failed to delete metric for labels %v. This may result in ambiguity from multiple metrics concurrently indicating different active configs.", activeConfigLabels)
	}
	// record the new timeseries
	activeConfigLabels = labels
	// expose the new timeseries with a constant count of 1
	ActiveConfig.With(activeConfigLabels).Set(1)
	return nil
}

// track labels across metric updates, so we can delete old label sets and prevent leaks
var lastKnownGoodConfigLabels map[string]string

// SetLastKnownGoodConfig tracks labels according to the NodeConfig that was successfully applied last.
// It also tracks labels across metric updates so old labels can be safely deleted.
func SetLastKnownGoodConfig(source *corev1.NodeConfigSource) error {
	// compute the timeseries labels from the source
	labels, err := configLabels(source)
	if err != nil {
		return err
	}
	// clean up the old timeseries (WithLabelValues creates a new one for each distinct label set)
	if !LastKnownGoodConfig.Delete(lastKnownGoodConfigLabels) {
		klog.Warningf("Failed to delete metric for labels %v. This may result in ambiguity from multiple metrics concurrently indicating different last known good configs.", lastKnownGoodConfigLabels)
	}
	// record the new timeseries
	lastKnownGoodConfigLabels = labels
	// expose the new timeseries with a constant count of 1
	LastKnownGoodConfig.With(lastKnownGoodConfigLabels).Set(1)
	return nil
}

// SetConfigError sets a the ConfigError metric to 1 in case any errors were encountered.
func SetConfigError(err bool) {
	if err {
		ConfigError.Set(1)
	} else {
		ConfigError.Set(0)
	}
}

// SetNodeName sets the NodeName Gauge to 1.
func SetNodeName(name types.NodeName) {
	NodeName.WithLabelValues(string(name)).Set(1)
}
