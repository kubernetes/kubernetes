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
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	k8sv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	kubestats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
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

	vol0          = "vol0"
	vol1          = "vol1"
	vol2          = "vol2"
	vol3          = "vol3"
	pvcClaimName0 = "pvc-fake0"
	pvcClaimName1 = "pvc-fake1"
)

var (
	ErrorWatchTimeout = errors.New("watch event timeout")
	// Create pod spec to test against
	podVolumes = []k8sv1.Volume{
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
					ClaimName: pvcClaimName0,
				},
			},
		},
		{
			Name: vol2,
			VolumeSource: k8sv1.VolumeSource{
				PersistentVolumeClaim: &k8sv1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvcClaimName1,
				},
			},
		},
		{
			Name: vol3,
			VolumeSource: k8sv1.VolumeSource{
				Ephemeral: &k8sv1.EphemeralVolumeSource{},
			},
		},
	}

	fakePod = &k8sv1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pName0,
			Namespace: namespace0,
			UID:       "UID" + pName0,
		},
		Spec: k8sv1.PodSpec{
			Volumes: podVolumes,
		},
	}
)

func TestPVCRef(t *testing.T) {
	ktesting.Init(t).SyncTest("", testPVCRef)
}

func testPVCRef(tCtx ktesting.TContext) {
	t := tCtx.TB()
	logger := tCtx.Logger()
	// Setup mock stats provider
	mockStats := statstest.NewMockProvider(t)
	volumes := map[string]volume.Volume{vol0: &fakeVolume{}, vol1: &fakeVolume{}, vol3: &fakeVolume{}}
	mockStats.EXPECT().ListVolumesForPod(fakePod.UID).Return(volumes, true)
	blockVolumes := map[string]volume.BlockVolume{vol2: &fakeBlockVolume{}}
	mockStats.EXPECT().ListBlockVolumesForPod(fakePod.UID).Return(blockVolumes, true)

	eventStore := make(chan string, 1)
	fakeEventRecorder := record.FakeRecorder{
		Events: eventStore,
	}

	// Calculate stats for pod
	statsCalculator := newVolumeStatCalculator(mockStats, time.Minute, fakePod, &fakeEventRecorder, nil)
	statsCalculator.calcAndStoreStats(logger)
	vs, _ := statsCalculator.GetLatest()

	assert.Len(t, append(vs.EphemeralVolumes, vs.PersistentVolumes...), 4)
	// Verify 'vol0' doesn't have a PVC reference
	assert.Contains(t, append(vs.EphemeralVolumes, vs.PersistentVolumes...), kubestats.VolumeStats{
		Name:    vol0,
		FsStats: expectedFSStats(),
	})
	// Verify 'vol1' has a PVC reference
	assert.Contains(t, append(vs.EphemeralVolumes, vs.PersistentVolumes...), kubestats.VolumeStats{
		Name: vol1,
		PVCRef: &kubestats.PVCReference{
			Name:      pvcClaimName0,
			Namespace: namespace0,
		},
		FsStats: expectedFSStats(),
	})
	// // Verify 'vol2' has a PVC reference
	assert.Contains(t, append(vs.EphemeralVolumes, vs.PersistentVolumes...), kubestats.VolumeStats{
		Name: vol2,
		PVCRef: &kubestats.PVCReference{
			Name:      pvcClaimName1,
			Namespace: namespace0,
		},
		FsStats: expectedBlockStats(),
	})
	// Verify 'vol3' has a PVC reference
	assert.Contains(t, append(vs.EphemeralVolumes, vs.PersistentVolumes...), kubestats.VolumeStats{
		Name: vol3,
		PVCRef: &kubestats.PVCReference{
			Name:      pName0 + "-" + vol3,
			Namespace: namespace0,
		},
		FsStats: expectedFSStats(),
	})
}

func TestNormalVolumeEvent(t *testing.T) {
	ktesting.Init(t).SyncTest("", testNormalVolumeEvent)
}

func testNormalVolumeEvent(tCtx ktesting.TContext) {
	t := tCtx.TB()
	logger := tCtx.Logger()
	mockStats := statstest.NewMockProvider(t)

	volumes := map[string]volume.Volume{vol0: &fakeVolume{}, vol1: &fakeVolume{}}
	mockStats.EXPECT().ListVolumesForPod(fakePod.UID).Return(volumes, true)
	blockVolumes := map[string]volume.BlockVolume{vol2: &fakeBlockVolume{}}
	mockStats.EXPECT().ListBlockVolumesForPod(fakePod.UID).Return(blockVolumes, true)

	eventStore := make(chan string, 2)
	fakeEventRecorder := record.FakeRecorder{
		Events: eventStore,
	}

	// Calculate stats for pod
	statsCalculator := newVolumeStatCalculator(mockStats, time.Minute, fakePod, &fakeEventRecorder, nil)
	statsCalculator.calcAndStoreStats(logger)

	event, err := WatchEvent(eventStore)
	assert.Error(t, err)
	assert.Equal(t, "", event)
}

func WatchEvent(eventChan <-chan string) (string, error) {
	select {
	case event := <-eventChan:
		return event, nil
	case <-time.After(5 * time.Second):
		return "", ErrorWatchTimeout
	}
}

// Fake volume/metrics provider
var _ volume.Volume = &fakeVolume{}

type fakeVolume struct{}

func (v *fakeVolume) GetPath() string { return "" }

func (v *fakeVolume) GetMetrics() (*volume.Metrics, error) {
	return expectedMetrics(), nil
}

func expectedMetrics() *volume.Metrics {
	vMetrics := &volume.Metrics{
		Available:  resource.NewQuantity(available, resource.BinarySI),
		Capacity:   resource.NewQuantity(capacity, resource.BinarySI),
		Used:       resource.NewQuantity(available-capacity, resource.BinarySI),
		Inodes:     resource.NewQuantity(inodesTotal, resource.BinarySI),
		InodesFree: resource.NewQuantity(inodesFree, resource.BinarySI),
		InodesUsed: resource.NewQuantity(inodesTotal-inodesFree, resource.BinarySI),
	}

	return vMetrics
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

// Fake block-volume/metrics provider, block-devices have no inodes
var _ volume.BlockVolume = &fakeBlockVolume{}

type fakeBlockVolume struct{}

func (v *fakeBlockVolume) GetGlobalMapPath(*volume.Spec) (string, error) { return "", nil }

func (v *fakeBlockVolume) GetPodDeviceMapPath() (string, string) { return "", "" }

func (v *fakeBlockVolume) SupportsMetrics() bool { return true }

func (v *fakeBlockVolume) GetMetrics() (*volume.Metrics, error) {
	return expectedBlockMetrics(), nil
}

func expectedBlockMetrics() *volume.Metrics {
	vMetrics := &volume.Metrics{
		Available: resource.NewQuantity(available, resource.BinarySI),
		Capacity:  resource.NewQuantity(capacity, resource.BinarySI),
		Used:      resource.NewQuantity(available-capacity, resource.BinarySI),
	}

	return vMetrics
}

func expectedBlockStats() kubestats.FsStats {
	metric := expectedBlockMetrics()
	available := uint64(metric.Available.Value())
	capacity := uint64(metric.Capacity.Value())
	used := uint64(metric.Used.Value())
	null := uint64(0)
	return kubestats.FsStats{
		AvailableBytes: &available,
		CapacityBytes:  &capacity,
		UsedBytes:      &used,
		Inodes:         &null,
		InodesFree:     &null,
		InodesUsed:     &null,
	}
}

func TestDynamicVolumeSpecRefresh(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicNodeLocalEphemeralVolumes, true)
	tCtx := ktesting.Init(t)
	logger := tCtx.Logger()

	initialPod := &k8sv1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pName0,
			Namespace: namespace0,
			UID:       "UID-dyn-vol",
		},
		Spec: k8sv1.PodSpec{
			Volumes: []k8sv1.Volume{
				{
					Name: vol0,
					VolumeSource: k8sv1.VolumeSource{
						EmptyDir: &k8sv1.EmptyDirVolumeSource{},
					},
				},
			},
		},
	}

	updatedPod := initialPod.DeepCopy()
	updatedPod.Spec.Volumes = append(updatedPod.Spec.Volumes, k8sv1.Volume{
		Name: "dyn-emptydir",
		VolumeSource: k8sv1.VolumeSource{
			EmptyDir: &k8sv1.EmptyDirVolumeSource{},
		},
	})

	t.Run("calcAndStoreStats refreshes pod spec when mounted volume is missing from cached spec", func(t *testing.T) {
		mockStats := statstest.NewMockProvider(t)
		volumes := map[string]volume.Volume{
			vol0:           &fakeVolume{},
			"dyn-emptydir": &fakeVolume{},
		}
		mockStats.EXPECT().ListVolumesForPod(initialPod.UID).Return(volumes, true)
		mockStats.EXPECT().ListBlockVolumesForPod(initialPod.UID).Return(nil, false)
		mockStats.EXPECT().GetPodByName(initialPod.Namespace, initialPod.Name).Return(updatedPod, true)

		statsCalculator := newVolumeStatCalculator(mockStats, time.Minute, initialPod, &record.FakeRecorder{}, nil)
		statsCalculator.calcAndStoreStats(logger)

		vs, ok := statsCalculator.GetLatest()
		require.True(t, ok)
		assert.Empty(t, vs.PersistentVolumes)
		assert.ElementsMatch(t, []kubestats.VolumeStats{
			{Name: vol0, FsStats: expectedFSStats()},
			{Name: "dyn-emptydir", FsStats: expectedFSStats()},
		}, vs.EphemeralVolumes)
	})

	t.Run("updateCachedPodVolumeStats refreshes cached calculator pod spec", func(t *testing.T) {
		mockStats := statstest.NewMockProvider(t)
		analyzer := newFsResourceAnalyzer(mockStats, 0, &record.FakeRecorder{}, nil)

		existingCalc := newVolumeStatCalculator(mockStats, time.Minute, initialPod, &record.FakeRecorder{}, nil)
		analyzer.cachedVolumeStats.Store(statCache{initialPod.UID: existingCalc})

		podWithConfigMap := updatedPod.DeepCopy()
		podWithConfigMap.Spec.Volumes = append(podWithConfigMap.Spec.Volumes, k8sv1.Volume{
			Name: "dyn-cm",
			VolumeSource: k8sv1.VolumeSource{
				ConfigMap: &k8sv1.ConfigMapVolumeSource{
					LocalObjectReference: k8sv1.LocalObjectReference{Name: "cm-1"},
				},
			},
		})
		mockStats.EXPECT().GetPods().Return([]*k8sv1.Pod{podWithConfigMap})
		analyzer.updateCachedPodVolumeStats(logger)

		volumes := map[string]volume.Volume{
			vol0:           &fakeVolume{},
			"dyn-emptydir": &fakeVolume{},
			"dyn-cm":       &fakeVolume{},
		}
		mockStats.EXPECT().ListVolumesForPod(initialPod.UID).Return(volumes, true)
		mockStats.EXPECT().ListBlockVolumesForPod(initialPod.UID).Return(nil, false)

		existingCalc.calcAndStoreStats(logger)
		vs, ok := analyzer.GetPodVolumeStats(initialPod.UID)
		require.True(t, ok)
		assert.Empty(t, vs.PersistentVolumes)
		assert.ElementsMatch(t, []kubestats.VolumeStats{
			{Name: vol0, FsStats: expectedFSStats()},
			{Name: "dyn-emptydir", FsStats: expectedFSStats()},
			{Name: "dyn-cm", FsStats: expectedFSStats()},
		}, vs.EphemeralVolumes)
	})

	t.Run("unmounting dynamic volume removed from allocated spec uses actuatedState", func(t *testing.T) {
		actuatedState := state.NewStateMemory(logger, nil, nil)
		podInActuated := initialPod.DeepCopy()
		podInActuated.Spec.Volumes = nil
		require.NoError(t, actuatedState.SetPod(logger, podInActuated))
		require.NoError(t, actuatedState.SetPodVolume(logger, initialPod.UID, updatedPod.Spec.Volumes[1]))

		// Allocated pod has already dropped dyn-emptydir, while ASW and actuatedState still hold it during teardown.
		mockStats := statstest.NewMockProvider(t)
		volumes := map[string]volume.Volume{
			vol0:           &fakeVolume{},
			"dyn-emptydir": &fakeVolume{},
		}
		mockStats.EXPECT().ListVolumesForPod(initialPod.UID).Return(volumes, true)
		mockStats.EXPECT().ListBlockVolumesForPod(initialPod.UID).Return(nil, false)

		statsCalculator := newVolumeStatCalculator(mockStats, time.Minute, initialPod, &record.FakeRecorder{}, actuatedState)
		statsCalculator.calcAndStoreStats(logger)

		vs, ok := statsCalculator.GetLatest()
		require.True(t, ok)
		assert.Empty(t, vs.PersistentVolumes)
		assert.ElementsMatch(t, []kubestats.VolumeStats{
			{Name: vol0, FsStats: expectedFSStats()},
			{Name: "dyn-emptydir", FsStats: expectedFSStats()},
		}, vs.EphemeralVolumes)
	})

	t.Run("actuatedState overrides pod spec when volume has same name with different source", func(t *testing.T) {
		actuatedState := state.NewStateMemory(logger, nil, nil)
		require.NoError(t, actuatedState.SetPodVolume(logger, initialPod.UID, k8sv1.Volume{
			Name: "readded-vol",
			VolumeSource: k8sv1.VolumeSource{
				EmptyDir: &k8sv1.EmptyDirVolumeSource{},
			},
		}))

		podWithReaddedSecret := initialPod.DeepCopy()
		podWithReaddedSecret.Spec.Volumes = append(podWithReaddedSecret.Spec.Volumes, k8sv1.Volume{
			Name: "readded-vol",
			VolumeSource: k8sv1.VolumeSource{
				Secret: &k8sv1.SecretVolumeSource{SecretName: "my-secret"},
			},
		})

		mockStats := statstest.NewMockProvider(t)
		volumes := map[string]volume.Volume{
			vol0:          &fakeVolume{},
			"readded-vol": &fakeVolume{},
		}
		mockStats.EXPECT().ListVolumesForPod(initialPod.UID).Return(volumes, true)
		mockStats.EXPECT().ListBlockVolumesForPod(initialPod.UID).Return(nil, false)

		statsCalculator := newVolumeStatCalculator(mockStats, time.Minute, podWithReaddedSecret, &record.FakeRecorder{}, actuatedState)
		statsCalculator.calcAndStoreStats(logger)

		vs, ok := statsCalculator.GetLatest()
		require.True(t, ok)
		assert.Empty(t, vs.PersistentVolumes)
		assert.ElementsMatch(t, []kubestats.VolumeStats{
			{Name: vol0, FsStats: expectedFSStats()},
			{Name: "readded-vol", FsStats: expectedFSStats()},
		}, vs.EphemeralVolumes)
	})

	t.Run("feature gate disabled does not refresh cached pod spec", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicNodeLocalEphemeralVolumes, false)

		mockStats := statstest.NewMockProvider(t)
		volumes := map[string]volume.Volume{
			vol0:           &fakeVolume{},
			"dyn-emptydir": &fakeVolume{},
		}
		mockStats.EXPECT().ListVolumesForPod(initialPod.UID).Return(volumes, true)
		mockStats.EXPECT().ListBlockVolumesForPod(initialPod.UID).Return(nil, false)

		statsCalculator := newVolumeStatCalculator(mockStats, time.Minute, initialPod, &record.FakeRecorder{}, nil)
		statsCalculator.calcAndStoreStats(logger)

		vs, ok := statsCalculator.GetLatest()
		require.True(t, ok)
		assert.Equal(t, []kubestats.VolumeStats{{Name: vol0, FsStats: expectedFSStats()}}, vs.EphemeralVolumes)
		assert.Equal(t, []kubestats.VolumeStats{{Name: "dyn-emptydir", FsStats: expectedFSStats()}}, vs.PersistentVolumes)
	})
}
