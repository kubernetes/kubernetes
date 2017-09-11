/*
Copyright 2017 The Kubernetes Authors.

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
	"github.com/stretchr/testify/assert"
	"testing"
	"time"

	k8sv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubestats "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	statstest "k8s.io/kubernetes/pkg/kubelet/server/stats/testing"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	namespace0  = "test0"
	pName0      = "pod0"
	capacity    = int64(10000000)
	available   = int64(5000000)
	inodesTotal = int64(2000)
	inodesFree  = int64(1000)

	vol0         = "vol0"
	vol1         = "vol1"
	pvcClaimName = "pvc-fake"
)

func TestPVCRef(t *testing.T) {
	// Create pod spec to test against
	podVolumes := []k8sv1.Volume{
		{
			Name: vol0,
			VolumeSource: k8sv1.VolumeSource{
				GCEPersistentDisk: &k8sv1.GCEPersistentDiskVolumeSource{
					PDName: "fake-device1",
				},
			},
		},
		{
			Name: vol1,
			VolumeSource: k8sv1.VolumeSource{
				PersistentVolumeClaim: &k8sv1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvcClaimName,
				},
			},
		},
	}

	fakePod := &k8sv1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pName0,
			Namespace: namespace0,
			UID:       "UID" + pName0,
		},
		Spec: k8sv1.PodSpec{
			Volumes: podVolumes,
		},
	}

	// Setup mock stats provider
	mockStats := new(statstest.StatsProvider)
	volumes := map[string]volume.Volume{vol0: &fakeVolume{}, vol1: &fakeVolume{}}
	mockStats.On("ListVolumesForPod", fakePod.UID).Return(volumes, true)

	// Calculate stats for pod
	statsCalculator := newVolumeStatCalculator(mockStats, time.Minute, fakePod)
	statsCalculator.calcAndStoreStats()
	vs, _ := statsCalculator.GetLatest()

	assert.Len(t, vs.Volumes, 2)
	// Verify 'vol0' doesn't have a PVC reference
	assert.Contains(t, vs.Volumes, kubestats.VolumeStats{
		Name:    vol0,
		FsStats: expectedFSStats(),
	})
	// Verify 'vol1' has a PVC reference
	assert.Contains(t, vs.Volumes, kubestats.VolumeStats{
		Name: vol1,
		PVCRef: &kubestats.PVCReference{
			Name:      pvcClaimName,
			Namespace: namespace0,
		},
		FsStats: expectedFSStats(),
	})
}

// Fake volume/metrics provider
var _ volume.Volume = &fakeVolume{}

type fakeVolume struct{}

func (v *fakeVolume) GetPath() string { return "" }

func (v *fakeVolume) GetMetrics() (*volume.Metrics, error) {
	return expectedMetrics(), nil
}

func expectedMetrics() *volume.Metrics {
	return &volume.Metrics{
		Available:  resource.NewQuantity(available, resource.BinarySI),
		Capacity:   resource.NewQuantity(capacity, resource.BinarySI),
		Used:       resource.NewQuantity(available-capacity, resource.BinarySI),
		Inodes:     resource.NewQuantity(inodesTotal, resource.BinarySI),
		InodesFree: resource.NewQuantity(inodesFree, resource.BinarySI),
		InodesUsed: resource.NewQuantity(inodesTotal-inodesFree, resource.BinarySI),
	}
}

func expectedFSStats() kubestats.FsStats {
	metric := expectedMetrics()
	available := uint64(metric.Available.Value())
	capacity := uint64(metric.Capacity.Value())
	used := uint64(metric.Used.Value())
	inodes := uint64(metric.Inodes.Value())
	inodesFree := uint64(metric.InodesFree.Value())
	inodesUsed := uint64(metric.InodesUsed.Value())
	return kubestats.FsStats{
		AvailableBytes: &available,
		CapacityBytes:  &capacity,
		UsedBytes:      &used,
		Inodes:         &inodes,
		InodesFree:     &inodesFree,
		InodesUsed:     &inodesUsed,
	}
}
