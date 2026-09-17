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
	"strings"
	"testing"

	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/utils/ptr"
)

func TestPartitionMetricsCollector(t *testing.T) {
	const (
		memoryUsageHeader = `
		# HELP kubelet_partition_memory_usage_bytes [ALPHA] Memory in bytes currently charged to the partition's cgroup, including its page cache.
		# TYPE kubelet_partition_memory_usage_bytes gauge
`
		podsHeader = `
		# HELP kubelet_partition_pods [ALPHA] Number of pods whose cgroup is in the partition. Only partitions configured on the node are reported, so an empty partition reports 0.
		# TYPE kubelet_partition_pods gauge
`
	)

	tests := []struct {
		name  string
		stats map[string]cm.PartitionStats
		want  string
	}{
		{
			name: "a partition reports its memory usage and pods",
			stats: map[string]cm.PartitionStats{
				"system": {MemoryUsageBytes: ptr.To[int64](1073741824), Pods: ptr.To[int64](3)},
			},
			want: memoryUsageHeader + `
		kubelet_partition_memory_usage_bytes{partition="system"} 1.073741824e+09
` + podsHeader + `
		kubelet_partition_pods{partition="system"} 3
`,
		},
		{
			name: "an empty partition reports zero pods",
			stats: map[string]cm.PartitionStats{
				"system": {MemoryUsageBytes: ptr.To[int64](0), Pods: ptr.To[int64](0)},
			},
			want: memoryUsageHeader + `
		kubelet_partition_memory_usage_bytes{partition="system"} 0
` + podsHeader + `
		kubelet_partition_pods{partition="system"} 0
`,
		},
		{
			name: "a stat that could not be read is left out",
			stats: map[string]cm.PartitionStats{
				"system": {Pods: ptr.To[int64](2)},
			},
			want: podsHeader + `
		kubelet_partition_pods{partition="system"} 2
`,
		},
		{
			// The node has only a system partition today, but the collector reports
			// one series per entry rather than a single hard-coded one.
			name: "every partition gets its own series",
			stats: map[string]cm.PartitionStats{
				"system": {MemoryUsageBytes: ptr.To[int64](1073741824), Pods: ptr.To[int64](3)},
				"other":  {MemoryUsageBytes: ptr.To[int64](536870912), Pods: ptr.To[int64](1)},
			},
			want: memoryUsageHeader + `
		kubelet_partition_memory_usage_bytes{partition="other"} 5.36870912e+08
		kubelet_partition_memory_usage_bytes{partition="system"} 1.073741824e+09
` + podsHeader + `
		kubelet_partition_pods{partition="other"} 1
		kubelet_partition_pods{partition="system"} 3
`,
		},
		{
			name:  "a node without a partition reports no series at all",
			stats: nil,
			want:  "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Refresh Desc to share with different registry
			descPartitionMemoryUsage = descPartitionMemoryUsage.GetRawDesc()
			descPartitionPods = descPartitionPods.GetRawDesc()

			collector := &partitionMetricsCollector{
				stats: func() map[string]cm.PartitionStats { return tc.stats },
			}

			err := testutil.CustomCollectAndCompare(collector, strings.NewReader(tc.want),
				"kubelet_partition_memory_usage_bytes", "kubelet_partition_pods")
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
