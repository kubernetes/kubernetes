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

	"k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	stats "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
)

// This file contains a series of deprecated metrics which we emit them by endpoint `/metrics/resource/v1alpha1`.
// These metrics have been adapted to new endpoint `/metrics/resource` as well as new `Desc`s.
// In general, we don't need to maintain these deprecated metrics any more.
// TODO(RainbowMango): Remove this file in release 1.20.0+.

var (
	nodeCPUUsageDesc = metrics.NewDesc("node_cpu_usage_seconds_total",
		"Cumulative cpu time consumed by the node in core-seconds",
		nil,
		nil,
		metrics.ALPHA,
		"1.18.0")

	nodeMemoryUsageDesc = metrics.NewDesc("node_memory_working_set_bytes",
		"Current working set of the node in bytes",
		nil,
		nil,
		metrics.ALPHA,
		"1.18.0")

	containerCPUUsageDesc = metrics.NewDesc("container_cpu_usage_seconds_total",
		"Cumulative cpu time consumed by the container in core-seconds",
		[]string{"container", "pod", "namespace"},
		nil,
		metrics.ALPHA,
		"1.18.0")

	containerMemoryUsageDesc = metrics.NewDesc("container_memory_working_set_bytes",
		"Current working set of the container in bytes",
		[]string{"container", "pod", "namespace"},
		nil,
		metrics.ALPHA,
		"1.18.0")
)

// getNodeCPUMetrics returns CPU utilization of a node.
func getNodeCPUMetrics(s stats.NodeStats) (*float64, time.Time) {
	if s.CPU == nil {
		return nil, time.Time{}
	}
	v := float64(*s.CPU.UsageCoreNanoSeconds) / float64(time.Second)
	return &v, s.CPU.Time.Time
}

// getNodeMemoryMetrics returns memory utilization of a node.
func getNodeMemoryMetrics(s stats.NodeStats) (*float64, time.Time) {
	if s.Memory == nil {
		return nil, time.Time{}
	}
	v := float64(*s.Memory.WorkingSetBytes)
	return &v, s.Memory.Time.Time
}

// getContainerCPUMetrics returns CPU utilization of a container.
func getContainerCPUMetrics(s stats.ContainerStats) (*float64, time.Time) {
	if s.CPU == nil {
		return nil, time.Time{}
	}
	v := float64(*s.CPU.UsageCoreNanoSeconds) / float64(time.Second)
	return &v, s.CPU.Time.Time
}

// getContainerMemoryMetrics returns memory utilization of a container.
func getContainerMemoryMetrics(s stats.ContainerStats) (*float64, time.Time) {
	if s.Memory == nil {
		return nil, time.Time{}
	}
	v := float64(*s.Memory.WorkingSetBytes)
	return &v, s.Memory.Time.Time
}

// NodeResourceMetric describes a metric for the node
type NodeResourceMetric struct {
	Desc    *metrics.Desc
	ValueFn func(stats.NodeStats) (*float64, time.Time)
}

func (n *NodeResourceMetric) desc() *metrics.Desc {
	return n.Desc
}

// ContainerResourceMetric describes a metric for containers
type ContainerResourceMetric struct {
	Desc    *metrics.Desc
	ValueFn func(stats.ContainerStats) (*float64, time.Time)
}

func (n *ContainerResourceMetric) desc() *metrics.Desc {
	return n.Desc
}

// ResourceMetricsConfig specifies which metrics to collect and export
type ResourceMetricsConfig struct {
	NodeMetrics      []NodeResourceMetric
	ContainerMetrics []ContainerResourceMetric
}

// Config is the v1alpha1 resource metrics definition
func Config() ResourceMetricsConfig {
	return ResourceMetricsConfig{
		NodeMetrics: []NodeResourceMetric{
			{
				Desc:    nodeCPUUsageDesc,
				ValueFn: getNodeCPUMetrics,
			},
			{
				Desc:    nodeMemoryUsageDesc,
				ValueFn: getNodeMemoryMetrics,
			},
		},
		ContainerMetrics: []ContainerResourceMetric{
			{
				Desc:    containerCPUUsageDesc,
				ValueFn: getContainerCPUMetrics,
			},
			{
				Desc:    containerMemoryUsageDesc,
				ValueFn: getContainerMemoryMetrics,
			},
		},
	}
}

// NewPrometheusResourceMetricCollector returns a metrics.StableCollector which exports resource metrics
func NewPrometheusResourceMetricCollector(provider SummaryProvider, config ResourceMetricsConfig) metrics.StableCollector {
	return &resourceMetricCollector{
		provider: provider,
		config:   config,
		errors: metrics.NewDesc("scrape_error",
			"1 if there was an error while getting container metrics, 0 otherwise",
			nil,
			nil,
			metrics.ALPHA,
			"1.18.0"),
	}
}

type resourceMetricCollector struct {
	metrics.BaseStableCollector

	provider SummaryProvider
	config   ResourceMetricsConfig
	errors   *metrics.Desc
}

var _ metrics.StableCollector = &resourceMetricCollector{}

// DescribeWithStability implements metrics.StableCollector
func (rc *resourceMetricCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- rc.errors

	for _, metric := range rc.config.NodeMetrics {
		ch <- metric.desc()
	}
	for _, metric := range rc.config.ContainerMetrics {
		ch <- metric.desc()
	}
}

// CollectWithStability implements metrics.StableCollector
// Since new containers are frequently created and removed, using the Gauge would
// leak metric collectors for containers or pods that no longer exist.  Instead, implement
// custom collector in a way that only collects metrics for active containers.
func (rc *resourceMetricCollector) CollectWithStability(ch chan<- metrics.Metric) {
	var errorCount float64
	defer func() {
		ch <- metrics.NewLazyConstMetric(rc.errors, metrics.GaugeValue, errorCount)
	}()
	summary, err := rc.provider.GetCPUAndMemoryStats()
	if err != nil {
		errorCount = 1
		klog.Warningf("Error getting summary for resourceMetric prometheus endpoint: %v", err)
		return
	}

	for _, metric := range rc.config.NodeMetrics {
		if value, timestamp := metric.ValueFn(summary.Node); value != nil {
			ch <- metrics.NewLazyMetricWithTimestamp(timestamp,
				metrics.NewLazyConstMetric(metric.desc(), metrics.GaugeValue, *value))
		}
	}

	for _, pod := range summary.Pods {
		for _, container := range pod.Containers {
			for _, metric := range rc.config.ContainerMetrics {
				if value, timestamp := metric.ValueFn(container); value != nil {
					ch <- metrics.NewLazyMetricWithTimestamp(timestamp,
						metrics.NewLazyConstMetric(metric.desc(), metrics.GaugeValue, *value, container.Name, pod.PodRef.Name, pod.PodRef.Namespace))
				}
			}
		}
	}
}
