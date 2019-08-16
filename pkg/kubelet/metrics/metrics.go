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

	"github.com/prometheus/client_golang/prometheus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// This const block defines the metric names for the kubelet metrics.
const (
	KubeletSubsystem                     = "kubelet"
	NodeNameKey                          = "node_name"
	NodeLabelKey                         = "node"
	PodWorkerDurationKey                 = "pod_worker_duration_seconds"
	PodStartDurationKey                  = "pod_start_duration_seconds"
	CgroupManagerOperationsKey           = "cgroup_manager_duration_seconds"
	PodWorkerStartDurationKey            = "pod_worker_start_duration_seconds"
	PLEGRelistDurationKey                = "pleg_relist_duration_seconds"
	PLEGDiscardEventsKey                 = "pleg_discard_events"
	PLEGRelistIntervalKey                = "pleg_relist_interval_seconds"
	EvictionsKey                         = "evictions"
	EvictionStatsAgeKey                  = "eviction_stats_age_seconds"
	DeprecatedPodWorkerLatencyKey        = "pod_worker_latency_microseconds"
	DeprecatedPodStartLatencyKey         = "pod_start_latency_microseconds"
	DeprecatedCgroupManagerOperationsKey = "cgroup_manager_latency_microseconds"
	DeprecatedPodWorkerStartLatencyKey   = "pod_worker_start_latency_microseconds"
	DeprecatedPLEGRelistLatencyKey       = "pleg_relist_latency_microseconds"
	DeprecatedPLEGRelistIntervalKey      = "pleg_relist_interval_microseconds"
	DeprecatedEvictionStatsAgeKey        = "eviction_stats_age_microseconds"
	VolumeStatsCapacityBytesKey          = "volume_stats_capacity_bytes"
	VolumeStatsAvailableBytesKey         = "volume_stats_available_bytes"
	VolumeStatsUsedBytesKey              = "volume_stats_used_bytes"
	VolumeStatsInodesKey                 = "volume_stats_inodes"
	VolumeStatsInodesFreeKey             = "volume_stats_inodes_free"
	VolumeStatsInodesUsedKey             = "volume_stats_inodes_used"
	// Metrics keys of remote runtime operations
	RuntimeOperationsKey                  = "runtime_operations_total"
	RuntimeOperationsDurationKey          = "runtime_operations_duration_seconds"
	RuntimeOperationsErrorsKey            = "runtime_operations_errors_total"
	DeprecatedRuntimeOperationsKey        = "runtime_operations"
	DeprecatedRuntimeOperationsLatencyKey = "runtime_operations_latency_microseconds"
	DeprecatedRuntimeOperationsErrorsKey  = "runtime_operations_errors"
	// Metrics keys of device plugin operations
	DevicePluginRegistrationCountKey           = "device_plugin_registration_total"
	DevicePluginAllocationDurationKey          = "device_plugin_alloc_duration_seconds"
	DeprecatedDevicePluginRegistrationCountKey = "device_plugin_registration_count"
	DeprecatedDevicePluginAllocationLatencyKey = "device_plugin_alloc_latency_microseconds"

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
	NodeName = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      NodeNameKey,
			Help:      "The node's name. The count is always 1.",
		},
		[]string{NodeLabelKey},
	)
	// ContainersPerPodCount is a Counter that tracks the number of containers per pod.
	ContainersPerPodCount = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      "containers_per_pod_count",
			Help:      "The number of containers per pod.",
			Buckets:   prometheus.DefBuckets,
		},
	)
	// PodWorkerDuration is a Histogram that tracks the duration (in seconds) in takes to sync a single pod.
	// Broken down by the operation type.
	PodWorkerDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      PodWorkerDurationKey,
			Help:      "Duration in seconds to sync a single pod. Broken down by operation type: create, update, or sync",
			Buckets:   prometheus.DefBuckets,
		},
		[]string{"operation_type"},
	)
	// PodStartDuration is a Histogram that tracks the duration (in seconds) it takes for a single pod to go from pending to running.
	PodStartDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      PodStartDurationKey,
			Help:      "Duration in seconds for a single pod to go from pending to running.",
			Buckets:   prometheus.DefBuckets,
		},
	)
	// CgroupManagerDuration is a Histogram that tracks the duration (in seconds) it takes for cgroup manager operations to complete.
	// Broken down by method.
	CgroupManagerDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      CgroupManagerOperationsKey,
			Help:      "Duration in seconds for cgroup manager operations. Broken down by method.",
			Buckets:   prometheus.DefBuckets,
		},
		[]string{"operation_type"},
	)
	// PodWorkerStartDuration is a Histogram that tracks the duration (in seconds) it takes from seeing a pod to starting a worker.
	PodWorkerStartDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      PodWorkerStartDurationKey,
			Help:      "Duration in seconds from seeing a pod to starting a worker.",
			Buckets:   prometheus.DefBuckets,
		},
	)
	// PLEGRelistDuration is a Histogram that tracks the duration (in seconds) it takes for relisting pods in the Kubelet's
	// Pod Lifecycle Event Generator (PLEG).
	PLEGRelistDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      PLEGRelistDurationKey,
			Help:      "Duration in seconds for relisting pods in PLEG.",
			Buckets:   prometheus.DefBuckets,
		},
	)
	// PLEGDiscardEvents is a Histogram that tracks the duration (in seconds) it takes for discarding events in the Kubelet's
	// Pod Lifecycle Event Generator (PLEG).
	PLEGDiscardEvents = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      PLEGDiscardEventsKey,
			Help:      "The number of discard events in PLEG.",
		},
		[]string{},
	)
	// PLEGRelistInterval is a Histogram that tracks the intervals (in seconds) between relisting in the Kubelet's
	// Pod Lifecycle Event Generator (PLEG).
	PLEGRelistInterval = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      PLEGRelistIntervalKey,
			Help:      "Interval in seconds between relisting in PLEG.",
			Buckets:   prometheus.DefBuckets,
		},
	)
	// RuntimeOperations is a Counter that tracks the cumulative number of remote runtime operations.
	// Broken down by operation type.
	RuntimeOperations = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      RuntimeOperationsKey,
			Help:      "Cumulative number of runtime operations by operation type.",
		},
		[]string{"operation_type"},
	)
	// RuntimeOperationsDuration is a Histogram that tracks the duration (in seconds) for remote runtime operations to complete.
	// Broken down by operation type.
	RuntimeOperationsDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      RuntimeOperationsDurationKey,
			Help:      "Duration in seconds of runtime operations. Broken down by operation type.",
			Buckets:   prometheus.DefBuckets,
		},
		[]string{"operation_type"},
	)
	// RuntimeOperationsErrors is a Counter that tracks the cumulative number of remote runtime operations errors.
	// Broken down by operation type.
	RuntimeOperationsErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      RuntimeOperationsErrorsKey,
			Help:      "Cumulative number of runtime operation errors by operation type.",
		},
		[]string{"operation_type"},
	)
	// Evictions is a Counter that tracks the cumulative number of pod evictions initiated by the kubelet.
	// Broken down by eviction signal.
	Evictions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      EvictionsKey,
			Help:      "Cumulative number of pod evictions by eviction signal",
		},
		[]string{"eviction_signal"},
	)
	// EvictionStatsAge is a Histogram that tracks the time (in seconds) between when stats are collected and when a pod is evicted
	// based on those stats. Broken down by eviction signal.
	EvictionStatsAge = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      EvictionStatsAgeKey,
			Help:      "Time between when stats are collected, and when pod is evicted based on those stats by eviction signal",
			Buckets:   prometheus.DefBuckets,
		},
		[]string{"eviction_signal"},
	)
	// DevicePluginRegistrationCount is a Counter that tracks the cumulative number of device plugin registrations.
	// Broken down by resource name.
	DevicePluginRegistrationCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      DevicePluginRegistrationCountKey,
			Help:      "Cumulative number of device plugin registrations. Broken down by resource name.",
		},
		[]string{"resource_name"},
	)
	// DevicePluginAllocationDuration is a Histogram that tracks the duration (in seconds) to serve a device plugin allocation request.
	// Broken down by resource name.
	DevicePluginAllocationDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      DevicePluginAllocationDurationKey,
			Help:      "Duration in seconds to serve a device plugin Allocation request. Broken down by resource name.",
			Buckets:   prometheus.DefBuckets,
		},
		[]string{"resource_name"},
	)
	// DeprecatedPodWorkerLatency is a Summary that tracks the latency (in microseconds) to sync a single pod.
	// Broken down by operation type. This metric is deprecated.
	DeprecatedPodWorkerLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedPodWorkerLatencyKey,
			Help:      "(Deprecated) Latency in microseconds to sync a single pod. Broken down by operation type: create, update, or sync",
		},
		[]string{"operation_type"},
	)
	// DeprecatedPodStartLatency is a Summary that tracks the latency (in microseconds) for a single pod to go from pending to running.
	// This metric is deprecated.
	DeprecatedPodStartLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedPodStartLatencyKey,
			Help:      "(Deprecated) Latency in microseconds for a single pod to go from pending to running.",
		},
	)
	// DeprecatedCgroupManagerLatency is a Summary that tracks the latency (in microseconds) for cgroup manager operations to complete.
	// Broken down by operation type. This metric is deprecated.
	DeprecatedCgroupManagerLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedCgroupManagerOperationsKey,
			Help:      "(Deprecated) Latency in microseconds for cgroup manager operations. Broken down by method.",
		},
		[]string{"operation_type"},
	)
	// DeprecatedPodWorkerStartLatency is a Summary that tracks the latency (in microseconds) from seeing a pod to starting a worker.
	// This metric is deprecated.
	DeprecatedPodWorkerStartLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedPodWorkerStartLatencyKey,
			Help:      "(Deprecated) Latency in microseconds from seeing a pod to starting a worker.",
		},
	)
	// DeprecatedPLEGRelistLatency is a Summary that tracks the latency (in microseconds) for relisting pods in PLEG.
	// This metric is deprecated.
	DeprecatedPLEGRelistLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedPLEGRelistLatencyKey,
			Help:      "(Deprecated) Latency in microseconds for relisting pods in PLEG.",
		},
	)
	// DeprecatedPLEGRelistInterval is a Summary that tracks the interval (in microseconds) between relistings in PLEG.
	// This metric is deprecated.
	DeprecatedPLEGRelistInterval = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedPLEGRelistIntervalKey,
			Help:      "(Deprecated) Interval in microseconds between relisting in PLEG.",
		},
	)
	// DeprecatedRuntimeOperations is a Counter that tracks the cumulative number of remote runtime operations.
	// Broken down by operation type. This metric is deprecated.
	DeprecatedRuntimeOperations = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedRuntimeOperationsKey,
			Help:      "(Deprecated) Cumulative number of runtime operations by operation type.",
		},
		[]string{"operation_type"},
	)
	// DeprecatedRuntimeOperationsLatency is a Summary that tracks the latency (in microseconds) of remote runtime operations
	// to complete. Broken down by operation type. This metric is deprecated.
	DeprecatedRuntimeOperationsLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedRuntimeOperationsLatencyKey,
			Help:      "(Deprecated) Latency in microseconds of runtime operations. Broken down by operation type.",
		},
		[]string{"operation_type"},
	)
	// DeprecatedRuntimeOperationsErrors is a Counter that tracks the cumulative number of remote runtime operation errors.
	// Broken down by operation type. This metric is deprecated.
	DeprecatedRuntimeOperationsErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedRuntimeOperationsErrorsKey,
			Help:      "(Deprecated) Cumulative number of runtime operation errors by operation type.",
		},
		[]string{"operation_type"},
	)
	// DeprecatedEvictionStatsAge is a Summary that tracks the time (in microseconds) between when stats are collected and when a pod
	// is evicted based on those stats. Broken down by eviction signal. This metric is deprecated.
	DeprecatedEvictionStatsAge = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedEvictionStatsAgeKey,
			Help:      "(Deprecated) Time between when stats are collected, and when pod is evicted based on those stats by eviction signal",
		},
		[]string{"eviction_signal"},
	)
	// DeprecatedDevicePluginRegistrationCount is a Counter that tracks the cumulative number of device plugin registrations.
	// Broken down by resource name. This metric is deprecated.
	DeprecatedDevicePluginRegistrationCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedDevicePluginRegistrationCountKey,
			Help:      "(Deprecated) Cumulative number of device plugin registrations. Broken down by resource name.",
		},
		[]string{"resource_name"},
	)
	// DeprecatedDevicePluginAllocationLatency is a Summary that tracks the latncy (in microseconds) for serving device plugin allocation requests.
	// Broken down by resource name. This metric is deprecated.
	DeprecatedDevicePluginAllocationLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      DeprecatedDevicePluginAllocationLatencyKey,
			Help:      "(Deprecated) Latency in microseconds to serve a device plugin Allocation request. Broken down by resource name.",
		},
		[]string{"resource_name"},
	)

	// Metrics for node config

	// AssignedConfig is a Gauge that is set 1 if the Kubelet has a NodeConfig assigned.
	AssignedConfig = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      AssignedConfigKey,
			Help:      "The node's understanding of intended config. The count is always 1.",
		},
		[]string{ConfigSourceLabelKey, ConfigUIDLabelKey, ConfigResourceVersionLabelKey, KubeletConfigKeyLabelKey},
	)
	// ActiveConfig is a Gauge that is set to 1 if the Kubelet has an active NodeConfig.
	ActiveConfig = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      ActiveConfigKey,
			Help:      "The config source the node is actively using. The count is always 1.",
		},
		[]string{ConfigSourceLabelKey, ConfigUIDLabelKey, ConfigResourceVersionLabelKey, KubeletConfigKeyLabelKey},
	)
	// LastKnownGoodConfig is a Gauge that is set to 1 if the Kubelet has a NodeConfig it can fall back to if there
	// are certain errors.
	LastKnownGoodConfig = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      LastKnownGoodConfigKey,
			Help:      "The config source the node will fall back to when it encounters certain errors. The count is always 1.",
		},
		[]string{ConfigSourceLabelKey, ConfigUIDLabelKey, ConfigResourceVersionLabelKey, KubeletConfigKeyLabelKey},
	)
	// ConfigError is a Gauge that is set to 1 if the node is experiencing a configuration-related error.
	ConfigError = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      ConfigErrorKey,
			Help:      "This metric is true (1) if the node is experiencing a configuration-related error, false (0) otherwise.",
		},
	)
	// RunPodSandboxDuration is a Histogram that tracks the duration (in seconds) it takes to run Pod Sandbox operations.
	// Broken down by RuntimeClass.
	RunPodSandboxDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: KubeletSubsystem,
			Name:      RunPodSandboxDurationKey,
			Help:      "Duration in seconds of the run_podsandbox operations. Broken down by RuntimeClass.",
			// Use DefBuckets for now, will customize the buckets if necessary.
			Buckets: prometheus.DefBuckets,
		},
		[]string{"runtime_handler"},
	)
	// RunPodSandboxErrors is a Counter that tracks the cumulative number of Pod Sandbox operations errors.
	// Broken down by RuntimeClass.
	RunPodSandboxErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      RunPodSandboxErrorsKey,
			Help:      "Cumulative number of the run_podsandbox operation errors by RuntimeClass.",
		},
		[]string{"runtime_handler"},
	)
)

var registerMetrics sync.Once

// Register registers all metrics.
func Register(containerCache kubecontainer.RuntimeCache, collectors ...prometheus.Collector) {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(NodeName)
		prometheus.MustRegister(PodWorkerDuration)
		prometheus.MustRegister(PodStartDuration)
		prometheus.MustRegister(CgroupManagerDuration)
		prometheus.MustRegister(PodWorkerStartDuration)
		prometheus.MustRegister(ContainersPerPodCount)
		prometheus.MustRegister(newPodAndContainerCollector(containerCache))
		prometheus.MustRegister(PLEGRelistDuration)
		prometheus.MustRegister(PLEGDiscardEvents)
		prometheus.MustRegister(PLEGRelistInterval)
		prometheus.MustRegister(RuntimeOperations)
		prometheus.MustRegister(RuntimeOperationsDuration)
		prometheus.MustRegister(RuntimeOperationsErrors)
		prometheus.MustRegister(Evictions)
		prometheus.MustRegister(EvictionStatsAge)
		prometheus.MustRegister(DevicePluginRegistrationCount)
		prometheus.MustRegister(DevicePluginAllocationDuration)
		prometheus.MustRegister(DeprecatedPodWorkerLatency)
		prometheus.MustRegister(DeprecatedPodStartLatency)
		prometheus.MustRegister(DeprecatedCgroupManagerLatency)
		prometheus.MustRegister(DeprecatedPodWorkerStartLatency)
		prometheus.MustRegister(DeprecatedPLEGRelistLatency)
		prometheus.MustRegister(DeprecatedPLEGRelistInterval)
		prometheus.MustRegister(DeprecatedRuntimeOperations)
		prometheus.MustRegister(DeprecatedRuntimeOperationsLatency)
		prometheus.MustRegister(DeprecatedRuntimeOperationsErrors)
		prometheus.MustRegister(DeprecatedEvictionStatsAge)
		prometheus.MustRegister(DeprecatedDevicePluginRegistrationCount)
		prometheus.MustRegister(DeprecatedDevicePluginAllocationLatency)
		if utilfeature.DefaultFeatureGate.Enabled(features.DynamicKubeletConfig) {
			prometheus.MustRegister(AssignedConfig)
			prometheus.MustRegister(ActiveConfig)
			prometheus.MustRegister(LastKnownGoodConfig)
			prometheus.MustRegister(ConfigError)
		}
		for _, collector := range collectors {
			prometheus.MustRegister(collector)
		}
	})
}

// SinceInMicroseconds gets the time since the specified start in microseconds.
func SinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}

func newPodAndContainerCollector(containerCache kubecontainer.RuntimeCache) *podAndContainerCollector {
	return &podAndContainerCollector{
		containerCache: containerCache,
	}
}

// Custom collector for current pod and container counts.
type podAndContainerCollector struct {
	// Cache for accessing information about running containers.
	containerCache kubecontainer.RuntimeCache
}

// TODO(vmarmol): Split by source?
var (
	runningPodCountDesc = prometheus.NewDesc(
		prometheus.BuildFQName("", KubeletSubsystem, "running_pod_count"),
		"Number of pods currently running",
		nil, nil)
	runningContainerCountDesc = prometheus.NewDesc(
		prometheus.BuildFQName("", KubeletSubsystem, "running_container_count"),
		"Number of containers currently running",
		nil, nil)
)

// Describe implements Prometheus' Describe method from the Collector interface. It sends all
// available descriptions to the provided channel and retunrs once the last description has been sent.
func (pc *podAndContainerCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- runningPodCountDesc
	ch <- runningContainerCountDesc
}

// Collect implements Prometheus' Collect method from the Collector interface. It's called by the Prometheus
// registry when collecting metrics.
func (pc *podAndContainerCollector) Collect(ch chan<- prometheus.Metric) {
	runningPods, err := pc.containerCache.GetPods()
	if err != nil {
		klog.Warningf("Failed to get running container information while collecting metrics: %v", err)
		return
	}

	runningContainers := 0
	for _, p := range runningPods {
		runningContainers += len(p.Containers)
	}
	ch <- prometheus.MustNewConstMetric(
		runningPodCountDesc,
		prometheus.GaugeValue,
		float64(len(runningPods)))
	ch <- prometheus.MustNewConstMetric(
		runningContainerCountDesc,
		prometheus.GaugeValue,
		float64(runningContainers))
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
