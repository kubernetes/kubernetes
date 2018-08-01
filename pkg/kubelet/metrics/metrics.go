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

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	corev1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

const (
	KubeletSubsystem             = "kubelet"
	PodWorkerLatencyKey          = "pod_worker_latency_microseconds"
	PodStartLatencyKey           = "pod_start_latency_microseconds"
	CgroupManagerOperationsKey   = "cgroup_manager_latency_microseconds"
	PodWorkerStartLatencyKey     = "pod_worker_start_latency_microseconds"
	PLEGRelistLatencyKey         = "pleg_relist_latency_microseconds"
	PLEGRelistIntervalKey        = "pleg_relist_interval_microseconds"
	EvictionStatsAgeKey          = "eviction_stats_age_microseconds"
	VolumeStatsCapacityBytesKey  = "volume_stats_capacity_bytes"
	VolumeStatsAvailableBytesKey = "volume_stats_available_bytes"
	VolumeStatsUsedBytesKey      = "volume_stats_used_bytes"
	VolumeStatsInodesKey         = "volume_stats_inodes"
	VolumeStatsInodesFreeKey     = "volume_stats_inodes_free"
	VolumeStatsInodesUsedKey     = "volume_stats_inodes_used"
	// Metrics keys of remote runtime operations
	RuntimeOperationsKey        = "runtime_operations"
	RuntimeOperationsLatencyKey = "runtime_operations_latency_microseconds"
	RuntimeOperationsErrorsKey  = "runtime_operations_errors"
	// Metrics keys of device plugin operations
	DevicePluginRegistrationCountKey = "device_plugin_registration_count"
	DevicePluginAllocationLatencyKey = "device_plugin_alloc_latency_microseconds"

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
)

var (
	ContainersPerPodCount = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      "containers_per_pod_count",
			Help:      "The number of containers per pod.",
		},
	)
	PodWorkerLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      PodWorkerLatencyKey,
			Help:      "Latency in microseconds to sync a single pod. Broken down by operation type: create, update, or sync",
		},
		[]string{"operation_type"},
	)
	PodStartLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      PodStartLatencyKey,
			Help:      "Latency in microseconds for a single pod to go from pending to running.",
		},
	)
	CgroupManagerLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      CgroupManagerOperationsKey,
			Help:      "Latency in microseconds for cgroup manager operations. Broken down by method.",
		},
		[]string{"operation_type"},
	)
	PodWorkerStartLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      PodWorkerStartLatencyKey,
			Help:      "Latency in microseconds from seeing a pod to starting a worker.",
		},
	)
	PLEGRelistLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      PLEGRelistLatencyKey,
			Help:      "Latency in microseconds for relisting pods in PLEG.",
		},
	)
	PLEGRelistInterval = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      PLEGRelistIntervalKey,
			Help:      "Interval in microseconds between relisting in PLEG.",
		},
	)
	// Metrics of remote runtime operations.
	RuntimeOperations = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      RuntimeOperationsKey,
			Help:      "Cumulative number of runtime operations by operation type.",
		},
		[]string{"operation_type"},
	)
	RuntimeOperationsLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      RuntimeOperationsLatencyKey,
			Help:      "Latency in microseconds of runtime operations. Broken down by operation type.",
		},
		[]string{"operation_type"},
	)
	RuntimeOperationsErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      RuntimeOperationsErrorsKey,
			Help:      "Cumulative number of runtime operation errors by operation type.",
		},
		[]string{"operation_type"},
	)
	EvictionStatsAge = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      EvictionStatsAgeKey,
			Help:      "Time between when stats are collected, and when pod is evicted based on those stats by eviction signal",
		},
		[]string{"eviction_signal"},
	)
	DevicePluginRegistrationCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      DevicePluginRegistrationCountKey,
			Help:      "Cumulative number of device plugin registrations. Broken down by resource name.",
		},
		[]string{"resource_name"},
	)
	DevicePluginAllocationLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      DevicePluginAllocationLatencyKey,
			Help:      "Latency in microseconds to serve a device plugin Allocation request. Broken down by resource name.",
		},
		[]string{"resource_name"},
	)

	// Metrics for node config

	AssignedConfig = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      AssignedConfigKey,
			Help:      "The node's understanding of intended config. The count is always 1.",
		},
		[]string{ConfigSourceLabelKey, ConfigUIDLabelKey, ConfigResourceVersionLabelKey, KubeletConfigKeyLabelKey},
	)
	ActiveConfig = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      ActiveConfigKey,
			Help:      "The config source the node is actively using. The count is always 1.",
		},
		[]string{ConfigSourceLabelKey, ConfigUIDLabelKey, ConfigResourceVersionLabelKey, KubeletConfigKeyLabelKey},
	)
	LastKnownGoodConfig = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      LastKnownGoodConfigKey,
			Help:      "The config source the node will fall back to when it encounters certain errors. The count is always 1.",
		},
		[]string{ConfigSourceLabelKey, ConfigUIDLabelKey, ConfigResourceVersionLabelKey, KubeletConfigKeyLabelKey},
	)
	ConfigError = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      ConfigErrorKey,
			Help:      "This metric is true (1) if the node is experiencing a configuration-related error, false (0) otherwise.",
		},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register(containerCache kubecontainer.RuntimeCache, collectors ...prometheus.Collector) {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(PodWorkerLatency)
		prometheus.MustRegister(PodStartLatency)
		prometheus.MustRegister(CgroupManagerLatency)
		prometheus.MustRegister(PodWorkerStartLatency)
		prometheus.MustRegister(ContainersPerPodCount)
		prometheus.MustRegister(newPodAndContainerCollector(containerCache))
		prometheus.MustRegister(PLEGRelistLatency)
		prometheus.MustRegister(PLEGRelistInterval)
		prometheus.MustRegister(RuntimeOperations)
		prometheus.MustRegister(RuntimeOperationsLatency)
		prometheus.MustRegister(RuntimeOperationsErrors)
		prometheus.MustRegister(EvictionStatsAge)
		prometheus.MustRegister(DevicePluginRegistrationCount)
		prometheus.MustRegister(DevicePluginAllocationLatency)
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

// Gets the time since the specified start in microseconds.
func SinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
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

func (pc *podAndContainerCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- runningPodCountDesc
	ch <- runningContainerCountDesc
}

func (pc *podAndContainerCollector) Collect(ch chan<- prometheus.Metric) {
	runningPods, err := pc.containerCache.GetPods()
	if err != nil {
		glog.Warningf("Failed to get running container information while collecting metrics: %v", err)
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
var assignedConfigLabels map[string]string = map[string]string{}

func SetAssignedConfig(source *corev1.NodeConfigSource) error {
	// compute the timeseries labels from the source
	labels, err := configLabels(source)
	if err != nil {
		return err
	}
	// clean up the old timeseries (WithLabelValues creates a new one for each distinct label set)
	AssignedConfig.Delete(assignedConfigLabels)
	// record the new timeseries
	assignedConfigLabels = labels
	// expose the new timeseries with a constant count of 1
	AssignedConfig.With(assignedConfigLabels).Set(1)
	return nil
}

// track labels across metric updates, so we can delete old label sets and prevent leaks
var activeConfigLabels map[string]string = map[string]string{}

func SetActiveConfig(source *corev1.NodeConfigSource) error {
	// compute the timeseries labels from the source
	labels, err := configLabels(source)
	if err != nil {
		return err
	}
	// clean up the old timeseries (WithLabelValues creates a new one for each distinct label set)
	ActiveConfig.Delete(activeConfigLabels)
	// record the new timeseries
	activeConfigLabels = labels
	// expose the new timeseries with a constant count of 1
	ActiveConfig.With(activeConfigLabels).Set(1)
	return nil
}

// track labels across metric updates, so we can delete old label sets and prevent leaks
var lastKnownGoodConfigLabels map[string]string = map[string]string{}

func SetLastKnownGoodConfig(source *corev1.NodeConfigSource) error {
	// compute the timeseries labels from the source
	labels, err := configLabels(source)
	if err != nil {
		return err
	}
	// clean up the old timeseries (WithLabelValues creates a new one for each distinct label set)
	LastKnownGoodConfig.Delete(lastKnownGoodConfigLabels)
	// record the new timeseries
	lastKnownGoodConfigLabels = labels
	// expose the new timeseries with a constant count of 1
	LastKnownGoodConfig.With(lastKnownGoodConfigLabels).Set(1)
	return nil
}

func SetConfigError(err bool) {
	if err {
		ConfigError.Set(1)
	} else {
		ConfigError.Set(0)
	}
}
