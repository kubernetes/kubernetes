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
	// TODO(vmarmol): Implement.
	// TODO(vmarmol): Split by source?
	PodCount = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: kubeletSubsystem,
			Name:      "pod_count",
			Help:      "Number of pods currently running.",
		},
	)
	// TODO(vmarmol): Implement.
	// TODO(vmarmol): Split by source?
	ContainerCount = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Subsystem: kubeletSubsystem,
			Name:      "container_count",
			Help:      "Number of containers currently running.",
		},
	)
	// TODO(vmarmol): Containers per pod
	// TODO(vmarmol): Latency of pod startup
	// TODO(vmarmol): Latency of SyncPods
)

func init() {
	// Register the metrics.
	prometheus.MustRegister(ImagePullLatency)
}
