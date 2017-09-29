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

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
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
			Help:      "Latency in microseconds for a single pod to go from pending to running. Broken down by podname.",
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
	VolumeStatsCapacityBytes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      VolumeStatsCapacityBytesKey,
			Help:      "Capacity in bytes of the volume",
		},
		[]string{"namespace", "persistentvolumeclaim"},
	)
	VolumeStatsAvailableBytes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      VolumeStatsAvailableBytesKey,
			Help:      "Number of available bytes in the volume",
		},
		[]string{"namespace", "persistentvolumeclaim"},
	)
	VolumeStatsUsedBytes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      VolumeStatsUsedBytesKey,
			Help:      "Number of used bytes in the volume",
		},
		[]string{"namespace", "persistentvolumeclaim"},
	)
	VolumeStatsInodes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      VolumeStatsInodesKey,
			Help:      "Maximum number of inodes in the volume",
		},
		[]string{"namespace", "persistentvolumeclaim"},
	)
	VolumeStatsInodesFree = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      VolumeStatsInodesFreeKey,
			Help:      "Number of free inodes in the volume",
		},
		[]string{"namespace", "persistentvolumeclaim"},
	)
	VolumeStatsInodesUsed = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: KubeletSubsystem,
			Name:      VolumeStatsInodesUsedKey,
			Help:      "Number of used inodes in the volume",
		},
		[]string{"namespace", "persistentvolumeclaim"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register(containerCache kubecontainer.RuntimeCache) {
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
		prometheus.MustRegister(VolumeStatsCapacityBytes)
		prometheus.MustRegister(VolumeStatsAvailableBytes)
		prometheus.MustRegister(VolumeStatsUsedBytes)
		prometheus.MustRegister(VolumeStatsInodes)
		prometheus.MustRegister(VolumeStatsInodesFree)
		prometheus.MustRegister(VolumeStatsInodesUsed)
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
