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
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

var (
	descPartitionMemoryUsage = metrics.NewDesc(
		"kubelet_partition_memory_usage_bytes",
		"Memory in bytes currently charged to the partition's cgroup, including its page cache.",
		[]string{"partition"}, nil,
		metrics.ALPHA,
		"",
	)
	descPartitionPods = metrics.NewDesc(
		"kubelet_partition_pods",
		"Number of pods whose cgroup is in the partition. Only partitions configured on the node are reported, so an empty partition reports 0.",
		[]string{"partition"}, nil,
		metrics.ALPHA,
		"",
	)
)

type partitionMetricsCollector struct {
	metrics.BaseStableCollector

	stats func() map[string]cm.PartitionStats
}

var _ metrics.StableCollector = &partitionMetricsCollector{}

// NewPartitionMetricsCollector exposes the usage of each node partition.
func NewPartitionMetricsCollector(stats func() map[string]cm.PartitionStats) metrics.StableCollector {
	return &partitionMetricsCollector{stats: stats}
}

// DescribeWithStability implements the metrics.StableCollector interface.
func (c *partitionMetricsCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- descPartitionMemoryUsage
	ch <- descPartitionPods
}

// CollectWithStability implements the metrics.StableCollector interface.
func (c *partitionMetricsCollector) CollectWithStability(ch chan<- metrics.Metric) {
	for partition, stats := range c.stats() {
		if stats.MemoryUsageBytes != nil {
			ch <- metrics.NewLazyConstMetric(descPartitionMemoryUsage, metrics.GaugeValue, float64(*stats.MemoryUsageBytes), partition)
		}
		if stats.Pods != nil {
			ch <- metrics.NewLazyConstMetric(descPartitionPods, metrics.GaugeValue, float64(*stats.Pods), partition)
		}
	}
}
