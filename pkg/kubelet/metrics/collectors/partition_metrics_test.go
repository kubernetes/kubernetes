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
)

func TestPartitionMetricsCollector(t *testing.T) {
	tests := []struct {
		name        string
		memoryUsage map[string]int64
		want        string
	}{
		{
			name:        "a partition reports its memory usage",
			memoryUsage: map[string]int64{"system": 1073741824},
			want: `
		# HELP kubelet_partition_memory_usage_bytes [ALPHA] Memory in bytes currently charged to the partition's cgroup, including its page cache.
		# TYPE kubelet_partition_memory_usage_bytes gauge
		kubelet_partition_memory_usage_bytes{partition="system"} 1.073741824e+09
`,
		},
		{
			// The node has only a system partition today, but the collector reports
			// one series per entry rather than a single hard-coded one.
			name: "every partition gets its own series",
			memoryUsage: map[string]int64{
				"system": 1073741824,
				"other":  536870912,
			},
			want: `
		# HELP kubelet_partition_memory_usage_bytes [ALPHA] Memory in bytes currently charged to the partition's cgroup, including its page cache.
		# TYPE kubelet_partition_memory_usage_bytes gauge
		kubelet_partition_memory_usage_bytes{partition="other"} 5.36870912e+08
		kubelet_partition_memory_usage_bytes{partition="system"} 1.073741824e+09
`,
		},
		{
			name:        "a node without a partition reports no series at all",
			memoryUsage: nil,
			want:        "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Refresh Desc to share with different registry
			descPartitionMemoryUsage = descPartitionMemoryUsage.GetRawDesc()

			collector := &partitionMetricsCollector{
				memoryUsage: func() map[string]int64 { return tc.memoryUsage },
			}

			err := testutil.CustomCollectAndCompare(collector, strings.NewReader(tc.want), "kubelet_partition_memory_usage_bytes")
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
