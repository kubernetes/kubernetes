/*
Copyright 2018 The Kubernetes Authors.

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

package collectors

import (
	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/klog"

	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
)

var (
	descLogSize = prometheus.NewDesc(
		"kubelet_container_log_filesystem_used_bytes",
		"Bytes used by the container's logs on the filesystem.",
		[]string{
			"namespace",
			"pod",
			"container",
		}, nil,
	)
)

type logMetricsCollector struct {
	podStats func() ([]statsapi.PodStats, error)
}

// NewLogMetricsCollector implements the prometheus.Collector interface and
// exposes metrics about container's log volume size.
func NewLogMetricsCollector(podStats func() ([]statsapi.PodStats, error)) prometheus.Collector {
	return &logMetricsCollector{
		podStats: podStats,
	}
}

// Describe implements the prometheus.Collector interface.
func (c *logMetricsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- descLogSize
}

// Collect implements the prometheus.Collector interface.
func (c *logMetricsCollector) Collect(ch chan<- prometheus.Metric) {
	podStats, err := c.podStats()
	if err != nil {
		klog.Errorf("failed to get pod stats: %v", err)
		return
	}

	for _, ps := range podStats {
		for _, c := range ps.Containers {
			if c.Logs != nil && c.Logs.UsedBytes != nil {
				ch <- prometheus.MustNewConstMetric(
					descLogSize,
					prometheus.GaugeValue,
					float64(*c.Logs.UsedBytes),
					ps.PodRef.Namespace,
					ps.PodRef.Name,
					c.Name,
				)
			}
		}
	}
}
