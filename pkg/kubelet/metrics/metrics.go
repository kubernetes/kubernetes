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

	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
)

const kubeletSubsystem = "kubelet"

var (
	ContainersPerPodCount = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: kubeletSubsystem,
			Name:      "containers_per_pod_count",
			Help:      "The number of containers per pod.",
		},
	)
	SyncPodLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: kubeletSubsystem,
			Name:      "sync_pod_latency_microseconds",
			Help:      "Latency in microseconds to sync a single pod. Broken down by operation type: create, update, or sync",
		},
		[]string{"operation_type"},
	)
	SyncPodsLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: kubeletSubsystem,
			Name:      "sync_pods_latency_microseconds",
			Help:      "Latency in microseconds to sync all pods.",
		},
	)
	DockerOperationsLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: kubeletSubsystem,
			Name:      "docker_operations_latency_microseconds",
			Help:      "Latency in microseconds of Docker operations. Broken down by operation type.",
		},
		[]string{"operation_type"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register(containerCache kubecontainer.RuntimeCache) {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(SyncPodLatency)
		prometheus.MustRegister(DockerOperationsLatency)
		prometheus.MustRegister(SyncPodsLatency)
		prometheus.MustRegister(ContainersPerPodCount)
		prometheus.MustRegister(newPodAndContainerCollector(containerCache))
	})
}

type SyncPodType int

const (
	SyncPodCreate SyncPodType = iota
	SyncPodUpdate
	SyncPodSync
)

func (sp SyncPodType) String() string {
	switch sp {
	case SyncPodCreate:
		return "create"
	case SyncPodUpdate:
		return "update"
	case SyncPodSync:
		return "sync"
	default:
		return "unknown"
	}
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
		prometheus.BuildFQName("", kubeletSubsystem, "running_pod_count"),
		"Number of pods currently running",
		nil, nil)
	runningContainerCountDesc = prometheus.NewDesc(
		prometheus.BuildFQName("", kubeletSubsystem, "running_container_count"),
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
		glog.Warning("Failed to get running container information while collecting metrics: %v", err)
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
