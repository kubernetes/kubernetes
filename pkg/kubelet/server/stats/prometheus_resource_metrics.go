/*
Copyright 2019 The Kubernetes Authors.

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

package stats

import (
	"time"

	"k8s.io/klog"
	stats "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"

	"github.com/prometheus/client_golang/prometheus"
)

// NodeResourceMetric describes a metric for the node
type NodeResourceMetric struct {
	Name        string
	Description string
	ValueFn     func(stats.NodeStats) (*float64, time.Time)
}

func (n *NodeResourceMetric) desc() *prometheus.Desc {
	return prometheus.NewDesc(n.Name, n.Description, []string{}, nil)
}

// ContainerResourceMetric describes a metric for containers
type ContainerResourceMetric struct {
	Name        string
	Description string
	ValueFn     func(stats.ContainerStats) (*float64, time.Time)
}

func (n *ContainerResourceMetric) desc() *prometheus.Desc {
	return prometheus.NewDesc(n.Name, n.Description, []string{"container", "pod", "namespace"}, nil)
}

// ResourceMetricsConfig specifies which metrics to collect and export
type ResourceMetricsConfig struct {
	NodeMetrics      []NodeResourceMetric
	ContainerMetrics []ContainerResourceMetric
}

// NewPrometheusResourceMetricCollector returns a prometheus.Collector which exports resource metrics
func NewPrometheusResourceMetricCollector(provider SummaryProvider, config ResourceMetricsConfig) prometheus.Collector {
	return &resourceMetricCollector{
		provider: provider,
		config:   config,
		errors: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "scrape_error",
			Help: "1 if there was an error while getting container metrics, 0 otherwise",
		}),
	}
}

type resourceMetricCollector struct {
	provider SummaryProvider
	config   ResourceMetricsConfig
	errors   prometheus.Gauge
}

var _ prometheus.Collector = &resourceMetricCollector{}

// Describe implements prometheus.Collector
func (rc *resourceMetricCollector) Describe(ch chan<- *prometheus.Desc) {
	rc.errors.Describe(ch)
	for _, metric := range rc.config.NodeMetrics {
		ch <- metric.desc()
	}
	for _, metric := range rc.config.ContainerMetrics {
		ch <- metric.desc()
	}
}

// Collect implements prometheus.Collector
// Since new containers are frequently created and removed, using the prometheus.Gauge Collector would
// leak metric collectors for containers or pods that no longer exist.  Instead, implement
// prometheus.Collector in a way that only collects metrics for active containers.
func (rc *resourceMetricCollector) Collect(ch chan<- prometheus.Metric) {
	rc.errors.Set(0)
	defer rc.errors.Collect(ch)
	summary, err := rc.provider.GetCPUAndMemoryStats()
	if err != nil {
		rc.errors.Set(1)
		klog.Warningf("Error getting summary for resourceMetric prometheus endpoint: %v", err)
		return
	}

	for _, metric := range rc.config.NodeMetrics {
		if value, timestamp := metric.ValueFn(summary.Node); value != nil {
			ch <- prometheus.NewMetricWithTimestamp(timestamp,
				prometheus.MustNewConstMetric(metric.desc(), prometheus.GaugeValue, *value))
		}
	}

	for _, pod := range summary.Pods {
		for _, container := range pod.Containers {
			for _, metric := range rc.config.ContainerMetrics {
				if value, timestamp := metric.ValueFn(container); value != nil {
					ch <- prometheus.NewMetricWithTimestamp(timestamp,
						prometheus.MustNewConstMetric(metric.desc(), prometheus.GaugeValue, *value, container.Name, pod.PodRef.Name, pod.PodRef.Namespace))
				}
			}
		}
	}
}
