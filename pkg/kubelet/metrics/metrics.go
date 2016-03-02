/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	KubeletSubsystem              = "kubelet"
	PodWorkerLatencyKey           = "pod_worker_latency_microseconds"
	SyncPodsLatencyKey            = "sync_pods_latency_microseconds"
	PodStartLatencyKey            = "pod_start_latency_microseconds"
	PodStatusLatencyKey           = "generate_pod_status_latency_microseconds"
	ContainerManagerOperationsKey = "container_manager_latency_microseconds"
	DockerOperationsKey           = "docker_operations_latency_microseconds"
	DockerErrorsKey               = "docker_errors"
	PodWorkerStartLatencyKey      = "pod_worker_start_latency_microseconds"
	PLEGRelistLatencyKey          = "pleg_relist_latency_microseconds"
	PLEGRelistIntervalKey         = "pleg_relist_interval_microseconds"
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
	SyncPodsLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      SyncPodsLatencyKey,
			Help:      "Latency in microseconds to sync all pods.",
		},
	)
	PodStartLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      PodStartLatencyKey,
			Help:      "Latency in microseconds for a single pod to go from pending to running. Broken down by podname.",
		},
	)
	PodStatusLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      PodStatusLatencyKey,
			Help:      "Latency in microseconds to generate status for a single pod.",
		},
	)
	ContainerManagerLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      ContainerManagerOperationsKey,
			Help:      "Latency in microseconds for container manager operations. Broken down by method.",
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
	DockerOperationsLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: KubeletSubsystem,
			Name:      DockerOperationsKey,
			Help:      "Latency in microseconds of Docker operations. Broken down by operation type.",
		},
		[]string{"operation_type"},
	)
	DockerErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: KubeletSubsystem,
			Name:      DockerErrorsKey,
			Help:      "Cumulative number of Docker errors by operation type.",
		},
		[]string{"operation_type"},
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
)

var registerMetrics sync.Once

// Register all metrics.
func Register(containerCache kubecontainer.RuntimeCache) {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(PodWorkerLatency)
		prometheus.MustRegister(PodStartLatency)
		prometheus.MustRegister(PodStatusLatency)
		prometheus.MustRegister(DockerOperationsLatency)
		prometheus.MustRegister(ContainerManagerLatency)
		prometheus.MustRegister(SyncPodsLatency)
		prometheus.MustRegister(PodWorkerStartLatency)
		prometheus.MustRegister(ContainersPerPodCount)
		prometheus.MustRegister(DockerErrors)
		prometheus.MustRegister(newPodAndContainerCollector(containerCache))
		prometheus.MustRegister(PLEGRelistLatency)
		prometheus.MustRegister(PLEGRelistInterval)
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
