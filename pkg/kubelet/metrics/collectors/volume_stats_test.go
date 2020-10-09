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
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-base/metrics/testutil"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	statstest "k8s.io/kubernetes/pkg/kubelet/server/stats/testing"
)

func newUint64Pointer(i uint64) *uint64 {
	return &i
}

func TestVolumeStatsCollector(t *testing.T) {
	// Fixed metadata on type and help text. We prepend this to every expected
	// output so we only have to modify a single place when doing adjustments.
	const metadata = `
		# HELP kubelet_volume_stats_available_bytes [ALPHA] Number of available bytes in the volume
		# TYPE kubelet_volume_stats_available_bytes gauge
		# HELP kubelet_volume_stats_capacity_bytes [ALPHA] Capacity in bytes of the volume
		# TYPE kubelet_volume_stats_capacity_bytes gauge
		# HELP kubelet_volume_stats_inodes [ALPHA] Maximum number of inodes in the volume
		# TYPE kubelet_volume_stats_inodes gauge
		# HELP kubelet_volume_stats_inodes_free [ALPHA] Number of free inodes in the volume
		# TYPE kubelet_volume_stats_inodes_free gauge
		# HELP kubelet_volume_stats_inodes_used [ALPHA] Number of used inodes in the volume
		# TYPE kubelet_volume_stats_inodes_used gauge
		# HELP kubelet_volume_stats_used_bytes [ALPHA] Number of used bytes in the volume
		# TYPE kubelet_volume_stats_used_bytes gauge
	`

	var (
		podStats = []statsapi.PodStats{
			{
				PodRef:    statsapi.PodReference{Name: "test-pod", Namespace: "test-namespace", UID: "UID_test-pod"},
				StartTime: metav1.Now(),
				VolumeStats: []statsapi.VolumeStats{
					{
						FsStats: statsapi.FsStats{
							Time:           metav1.Now(),
							AvailableBytes: newUint64Pointer(5.663154176e+09),
							CapacityBytes:  newUint64Pointer(1.0434699264e+10),
							UsedBytes:      newUint64Pointer(4.21789696e+09),
							InodesFree:     newUint64Pointer(655344),
							Inodes:         newUint64Pointer(655360),
							InodesUsed:     newUint64Pointer(16),
						},
						Name:   "test",
						PVCRef: nil,
					},
					{
						FsStats: statsapi.FsStats{
							Time:           metav1.Now(),
							AvailableBytes: newUint64Pointer(5.663154176e+09),
							CapacityBytes:  newUint64Pointer(1.0434699264e+10),
							UsedBytes:      newUint64Pointer(4.21789696e+09),
							InodesFree:     newUint64Pointer(655344),
							Inodes:         newUint64Pointer(655360),
							InodesUsed:     newUint64Pointer(16),
						},
						Name: "test",
						PVCRef: &statsapi.PVCReference{
							Name:      "testpvc",
							Namespace: "testns",
						},
					},
				},
			},
			{
				// Another pod references the same PVC (test-namespace/testpvc).
				PodRef:    statsapi.PodReference{Name: "test-pod-2", Namespace: "test-namespace", UID: "UID_test-pod"},
				StartTime: metav1.Now(),
				VolumeStats: []statsapi.VolumeStats{
					{
						FsStats: statsapi.FsStats{
							Time:           metav1.Now(),
							AvailableBytes: newUint64Pointer(5.663154176e+09),
							CapacityBytes:  newUint64Pointer(1.0434699264e+10),
							UsedBytes:      newUint64Pointer(4.21789696e+09),
							InodesFree:     newUint64Pointer(655344),
							Inodes:         newUint64Pointer(655360),
							InodesUsed:     newUint64Pointer(16),
						},
						Name: "test",
						PVCRef: &statsapi.PVCReference{
							Name:      "testpvc",
							Namespace: "testns",
						},
					},
				},
			},
		}

		want = metadata + `
			kubelet_volume_stats_available_bytes{namespace="testns",persistentvolumeclaim="testpvc"} 5.663154176e+09
			kubelet_volume_stats_capacity_bytes{namespace="testns",persistentvolumeclaim="testpvc"} 1.0434699264e+10
			kubelet_volume_stats_inodes{namespace="testns",persistentvolumeclaim="testpvc"} 655360
			kubelet_volume_stats_inodes_free{namespace="testns",persistentvolumeclaim="testpvc"} 655344
			kubelet_volume_stats_inodes_used{namespace="testns",persistentvolumeclaim="testpvc"} 16
			kubelet_volume_stats_used_bytes{namespace="testns",persistentvolumeclaim="testpvc"} 4.21789696e+09
			`

		metrics = []string{
			"kubelet_volume_stats_available_bytes",
			"kubelet_volume_stats_capacity_bytes",
			"kubelet_volume_stats_inodes",
			"kubelet_volume_stats_inodes_free",
			"kubelet_volume_stats_inodes_used",
			"kubelet_volume_stats_used_bytes",
		}
	)

	mockStatsProvider := new(statstest.StatsProvider)
	mockStatsProvider.On("ListPodStats").Return(podStats, nil)
	mockStatsProvider.On("ListPodStatsAndUpdateCPUNanoCoreUsage").Return(podStats, nil)
	if err := testutil.CustomCollectAndCompare(&volumeStatsCollector{statsProvider: mockStatsProvider}, strings.NewReader(want), metrics...); err != nil {
		t.Errorf("unexpected collecting result:\n%s", err)
	}
}
