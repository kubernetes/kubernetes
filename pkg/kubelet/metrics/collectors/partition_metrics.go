/*
Copyright The Kubernetes Authors.

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
	"k8s.io/component-base/metrics"
)

var descPartitionMemoryUsage = metrics.NewDesc(
	"kubelet_partition_memory_usage_bytes",
	"Memory in bytes currently charged to the partition's cgroup, including its page cache.",
	[]string{"partition"}, nil,
	metrics.ALPHA,
	"",
)

type partitionMetricsCollector struct {
	metrics.BaseStableCollector

	memoryUsage func() map[string]int64
}

var _ metrics.StableCollector = &partitionMetricsCollector{}

// NewPartitionMetricsCollector exposes the memory each node partition is using.
func NewPartitionMetricsCollector(memoryUsage func() map[string]int64) metrics.StableCollector {
	return &partitionMetricsCollector{memoryUsage: memoryUsage}
}

// DescribeWithStability implements the metrics.StableCollector interface.
func (c *partitionMetricsCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- descPartitionMemoryUsage
}

// CollectWithStability implements the metrics.StableCollector interface.
func (c *partitionMetricsCollector) CollectWithStability(ch chan<- metrics.Metric) {
	for partition, usage := range c.memoryUsage() {
		ch <- metrics.NewLazyConstMetric(descPartitionMemoryUsage, metrics.GaugeValue, float64(usage), partition)
	}
}
