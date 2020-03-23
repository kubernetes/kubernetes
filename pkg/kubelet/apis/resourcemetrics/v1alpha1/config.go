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

package v1alpha1

import (
	"time"

	"k8s.io/component-base/metrics"
	summary "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
)

// This file contains a series of deprecated metrics which we emit them by endpoint `/metrics/resource/v1alpha1`.
// These metrics have been adapted to new endpoint `/metrics/resource` as well as new `Desc`s.
// In general, we don't need to maintain these deprecated metrics any more.
// TODO(RainbowMango): Remove this file in release 1.20.0+.

// Version is the string representation of the version of this configuration
const Version = "v1alpha1"

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
func getNodeCPUMetrics(s summary.NodeStats) (*float64, time.Time) {
	if s.CPU == nil {
		return nil, time.Time{}
	}
	v := float64(*s.CPU.UsageCoreNanoSeconds) / float64(time.Second)
	return &v, s.CPU.Time.Time
}

// getNodeMemoryMetrics returns memory utilization of a node.
func getNodeMemoryMetrics(s summary.NodeStats) (*float64, time.Time) {
	if s.Memory == nil {
		return nil, time.Time{}
	}
	v := float64(*s.Memory.WorkingSetBytes)
	return &v, s.Memory.Time.Time
}

// getContainerCPUMetrics returns CPU utilization of a container.
func getContainerCPUMetrics(s summary.ContainerStats) (*float64, time.Time) {
	if s.CPU == nil {
		return nil, time.Time{}
	}
	v := float64(*s.CPU.UsageCoreNanoSeconds) / float64(time.Second)
	return &v, s.CPU.Time.Time
}

// getContainerMemoryMetrics returns memory utilization of a container.
func getContainerMemoryMetrics(s summary.ContainerStats) (*float64, time.Time) {
	if s.Memory == nil {
		return nil, time.Time{}
	}
	v := float64(*s.Memory.WorkingSetBytes)
	return &v, s.Memory.Time.Time
}

// Config is the v1alpha1 resource metrics definition
func Config() stats.ResourceMetricsConfig {
	return stats.ResourceMetricsConfig{
		NodeMetrics: []stats.NodeResourceMetric{
			{
				Desc:    nodeCPUUsageDesc,
				ValueFn: getNodeCPUMetrics,
			},
			{
				Desc:    nodeMemoryUsageDesc,
				ValueFn: getNodeMemoryMetrics,
			},
		},
		ContainerMetrics: []stats.ContainerResourceMetric{
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
