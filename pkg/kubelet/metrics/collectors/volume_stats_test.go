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
	"context"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-base/metrics/testutil"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	statstest "k8s.io/kubernetes/pkg/kubelet/server/stats/testing"
	"k8s.io/utils/ptr"
)

func TestVolumeStatsCollector(t *testing.T) {
	ctx := context.Background()
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
		# HELP kubelet_volume_stats_health_status_abnormal [ALPHA] Abnormal volume health status. The count is either 1 or 0. 1 indicates the volume is unhealthy, 0 indicates volume is healthy
		# TYPE kubelet_volume_stats_health_status_abnormal gauge
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
							AvailableBytes: ptr.To[uint64](5.663154176e+09),
							CapacityBytes:  ptr.To[uint64](1.0434699264e+10),
							UsedBytes:      ptr.To[uint64](4.21789696e+09),
							InodesFree:     ptr.To[uint64](655344),
							Inodes:         ptr.To[uint64](655360),
							InodesUsed:     ptr.To[uint64](16),
						},
						Name:   "test",
						PVCRef: nil,
					},
					{
						FsStats: statsapi.FsStats{
							Time:           metav1.Now(),
							AvailableBytes: ptr.To[uint64](5.663154176e+09),
							CapacityBytes:  ptr.To[uint64](1.0434699264e+10),
							UsedBytes:      ptr.To[uint64](4.21789696e+09),
							InodesFree:     ptr.To[uint64](655344),
							Inodes:         ptr.To[uint64](655360),
							InodesUsed:     ptr.To[uint64](16),
						},
						Name: "test",
						PVCRef: &statsapi.PVCReference{
							Name:      "testpvc",
							Namespace: "testns",
						},
						VolumeHealthStats: &statsapi.VolumeHealthStats{
							Abnormal: true,
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
							AvailableBytes: ptr.To[uint64](5.663154176e+09),
							CapacityBytes:  ptr.To[uint64](1.0434699264e+10),
							UsedBytes:      ptr.To[uint64](4.21789696e+09),
							InodesFree:     ptr.To[uint64](655344),
							Inodes:         ptr.To[uint64](655360),
							InodesUsed:     ptr.To[uint64](16),
						},
						Name: "test",
						PVCRef: &statsapi.PVCReference{
							Name:      "testpvc",
							Namespace: "testns",
						},
						VolumeHealthStats: &statsapi.VolumeHealthStats{
							Abnormal: true,
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
			kubelet_volume_stats_health_status_abnormal{namespace="testns",persistentvolumeclaim="testpvc"} 1
			`

		metrics = []string{
			"kubelet_volume_stats_available_bytes",
			"kubelet_volume_stats_capacity_bytes",
			"kubelet_volume_stats_inodes",
			"kubelet_volume_stats_inodes_free",
			"kubelet_volume_stats_inodes_used",
			"kubelet_volume_stats_used_bytes",
			"kubelet_volume_stats_health_status_abnormal",
		}
	)

	mockStatsProvider := statstest.NewMockProvider(t)

	mockStatsProvider.EXPECT().ListPodStats(ctx).Return(podStats, nil).Maybe()
	mockStatsProvider.EXPECT().ListPodStatsAndUpdateCPUNanoCoreUsage(ctx).Return(podStats, nil).Maybe()
	if err := testutil.CustomCollectAndCompare(&volumeStatsCollector{statsProvider: mockStatsProvider}, strings.NewReader(want), metrics...); err != nil {
		t.Errorf("unexpected collecting result:\n%s", err)
	}
}

func TestVolumeStatsCollectorWithNullVolumeStatus(t *testing.T) {
	ctx := context.Background()
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
							AvailableBytes: ptr.To[uint64](5.663154176e+09),
							CapacityBytes:  ptr.To[uint64](1.0434699264e+10),
							UsedBytes:      ptr.To[uint64](4.21789696e+09),
							InodesFree:     ptr.To[uint64](655344),
							Inodes:         ptr.To[uint64](655360),
							InodesUsed:     ptr.To[uint64](16),
						},
						Name:   "test",
						PVCRef: nil,
					},
					{
						FsStats: statsapi.FsStats{
							Time:           metav1.Now(),
							AvailableBytes: ptr.To[uint64](5.663154176e+09),
							CapacityBytes:  ptr.To[uint64](1.0434699264e+10),
							UsedBytes:      ptr.To[uint64](4.21789696e+09),
							InodesFree:     ptr.To[uint64](655344),
							Inodes:         ptr.To[uint64](655360),
							InodesUsed:     ptr.To[uint64](16),
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

	mockStatsProvider := statstest.NewMockProvider(t)

	mockStatsProvider.EXPECT().ListPodStats(ctx).Return(podStats, nil).Maybe()
	mockStatsProvider.EXPECT().ListPodStatsAndUpdateCPUNanoCoreUsage(ctx).Return(podStats, nil).Maybe()
	if err := testutil.CustomCollectAndCompare(&volumeStatsCollector{statsProvider: mockStatsProvider}, strings.NewReader(want), metrics...); err != nil {
		t.Errorf("unexpected collecting result:\n%s", err)
	}
}
