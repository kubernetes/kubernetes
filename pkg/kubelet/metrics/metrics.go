/*
Copyright 2015 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
)

const kubeletSubsystem = "kubelet"

var (
	ImagePullLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Subsystem: kubeletSubsystem,
			Name:      "image_pull_latency_microseconds",
			Help:      "Image pull latency in microseconds.",
		},
	)
	// TODO(vmarmol): Containers per pod
	// TODO(vmarmol): Latency of pod startup
	// TODO(vmarmol): Latency of SyncPods
)

var registerMetrics sync.Once

// Register all metrics.
func Register(containerCache dockertools.DockerCache) {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(ImagePullLatency)
		prometheus.MustRegister(newPodAndContainerCollector(containerCache))
	})
}

func newPodAndContainerCollector(containerCache dockertools.DockerCache) *podAndContainerCollector {
	return &podAndContainerCollector{
		containerCache: containerCache,
	}
}

// Custom collector for current pod and container counts.
type podAndContainerCollector struct {
	// Cache for accessing information about running containers.
	containerCache dockertools.DockerCache
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

func (self *podAndContainerCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- runningPodCountDesc
	ch <- runningContainerCountDesc
}

func (self *podAndContainerCollector) Collect(ch chan<- prometheus.Metric) {
	runningContainers, err := self.containerCache.RunningContainers()
	if err != nil {
		glog.Warning("Failed to get running container information while collecting metrics: %v", err)
		return
	}

	// Get a mapping of pod to number of containers in that pod.
	podToContainerCount := make(map[types.UID]struct{})
	for _, cont := range runningContainers {
		_, uid, _, _ := dockertools.ParseDockerName(cont.Names[0])
		podToContainerCount[uid] = struct{}{}
	}

	ch <- prometheus.MustNewConstMetric(
		runningPodCountDesc,
		prometheus.GaugeValue,
		float64(len(podToContainerCount)))
	ch <- prometheus.MustNewConstMetric(
		runningContainerCountDesc,
		prometheus.GaugeValue,
		float64(len(runningContainers)))
}
