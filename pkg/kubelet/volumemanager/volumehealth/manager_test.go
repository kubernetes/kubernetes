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

package volumehealth

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
)

type testStatusUpdater struct {
	mu    sync.Mutex
	calls []statusCall
}

type statusCall struct {
	podUID     types.UID
	volumeName string
	conditions []v1.VolumeHealthCondition
}

func (t *testStatusUpdater) SetPodVolumeHealth(logger klog.Logger, podUID types.UID, volumeName string, conditions []v1.VolumeHealthCondition) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.calls = append(t.calls, statusCall{
		podUID:     podUID,
		volumeName: volumeName,
		conditions: append([]v1.VolumeHealthCondition(nil), conditions...),
	})
	return true
}

func (t *testStatusUpdater) callCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.calls)
}

type fakeHealthClient struct {
	supportsVolumeHealth  bool
	supportsStorageHealth bool
	volumeConditions      []v1.VolumeHealthCondition
	storageConditions     []storagev1.StorageHealthCondition
	volumeErr             error
	storageErr            error
	volumeCalls           int
	storageCalls          int
}

func (f *fakeHealthClient) NodeGetVolumeHealth(ctx context.Context, volID, stagingTargetPath, volumePublishPath string) ([]v1.VolumeHealthCondition, error) {
	f.volumeCalls++
	if f.volumeErr != nil {
		return nil, f.volumeErr
	}
	return f.volumeConditions, nil
}

func (f *fakeHealthClient) NodeGetStorageHealth(ctx context.Context, secrets map[string]string) ([]storagev1.StorageHealthCondition, error) {
	f.storageCalls++
	if f.storageErr != nil {
		return nil, f.storageErr
	}
	return f.storageConditions, nil
}

func (f *fakeHealthClient) NodeSupportsVolumeHealth(ctx context.Context) (bool, error) {
	return f.supportsVolumeHealth, nil
}

func (f *fakeHealthClient) NodeSupportsStorageHealth(ctx context.Context) (bool, error) {
	return f.supportsStorageHealth, nil
}

type fakeCSINodeUpdater struct {
	mu    sync.Mutex
	calls []storageCall
}

type storageCall struct {
	driverName string
	conditions []storagev1.StorageHealthCondition
}

func (f *fakeCSINodeUpdater) UpdateCSINodeStorageHealth(driverName string, conditions []storagev1.StorageHealthCondition) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, storageCall{
		driverName: driverName,
		conditions: append([]storagev1.StorageHealthCondition(nil), conditions...),
	})
	return nil
}

func newTestManager(t *testing.T, statusUpdater StatusUpdater, client csi.HealthClient) *manager {
	t.Helper()
	logger, _ := ktesting.NewTestContext(t)
	volumePluginMgr, _ := volumetesting.GetTestKubeletVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr, util.NewFakeSELinuxLabelTranslator())
	asw := cache.NewActualStateOfWorld("node1", volumePluginMgr)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: "pod-uid-1", Namespace: "ns"},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: "pvc1"},
				},
			}},
		},
	}
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "pv1"},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:       "test.driver",
					VolumeHandle: "handle-1",
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		},
	}
	volumeSpec := volume.NewSpecFromPersistentVolume(pv, false)
	volumeName, err := dsw.AddPodToVolume(logger, util.GetUniquePodName(pod), pod, volumeSpec, "vol1", "", nil)
	if err != nil {
		t.Fatalf("AddPodToVolume: %v", err)
	}
	asw.MarkVolumeAsMountAttempted(util.GetUniquePodName(pod), volumeName)

	return &manager{
		dsw:           dsw,
		asw:           asw,
		statusUpdater: statusUpdater,
		probeInterval: time.Minute,
		clientFactory: func(driverName string) csi.HealthClient {
			return client
		},
		listDrivers: func() []string { return []string{"test.driver"} },
		csiNodeUpdater: func() CSINodeUpdater {
			return nil
		},
	}
}

func TestProbeVolumeHealth(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, true)

	degraded := []v1.VolumeHealthCondition{{
		Status: v1.VolumeHealthDegraded, Reason: "DiskSlow", Message: "slow disk",
	}}

	type step struct {
		volumeConditions []v1.VolumeHealthCondition
		volumeErr        error
		wantCallCount    int
	}

	tests := []struct {
		name                 string
		supportsVolumeHealth bool
		steps                []step
	}{
		{
			name:                 "bad health updates pod status",
			supportsVolumeHealth: true,
			steps: []step{
				{volumeConditions: degraded, wantCallCount: 1},
			},
		},
		{
			name:                 "cleared health clears pod status",
			supportsVolumeHealth: true,
			steps: []step{
				{volumeConditions: degraded, wantCallCount: 1},
				{volumeConditions: nil, wantCallCount: 2},
			},
		},
		{
			// Manager always forwards; status_manager suppresses no-op PATCHes.
			name:                 "same condition twice still calls status updater",
			supportsVolumeHealth: true,
			steps: []step{
				{volumeConditions: degraded, wantCallCount: 1},
				{volumeConditions: degraded, wantCallCount: 2},
			},
		},
		{
			name:                 "error leaves previous conditions unchanged",
			supportsVolumeHealth: true,
			steps: []step{
				{volumeConditions: degraded, wantCallCount: 1},
				{volumeErr: errors.New("rpc failed"), wantCallCount: 1},
			},
		},
		{
			name:                 "capability not supported skips probe",
			supportsVolumeHealth: false,
			steps: []step{
				{wantCallCount: 0},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			status := &testStatusUpdater{}
			client := &fakeHealthClient{supportsVolumeHealth: tc.supportsVolumeHealth}
			m := newTestManager(t, status, client)

			prevCalls := 0
			for i, s := range tc.steps {
				client.volumeConditions = s.volumeConditions
				client.volumeErr = s.volumeErr
				m.probeVolumeHealth(context.Background())

				got := status.callCount()
				if got != s.wantCallCount {
					t.Fatalf("step %d: expected %d status update(s), got %d", i, s.wantCallCount, got)
				}
				if got > prevCalls {
					last := status.calls[got-1]
					if last.podUID != "pod-uid-1" || last.volumeName != "vol1" {
						t.Fatalf("step %d: update for wrong pod/volume: %+v", i, last)
					}
					if len(last.conditions) != len(s.volumeConditions) {
						t.Fatalf("step %d: expected %d conditions, got %d", i, len(s.volumeConditions), len(last.conditions))
					}
					for j, want := range s.volumeConditions {
						if last.conditions[j].Status != want.Status || last.conditions[j].Reason != want.Reason {
							t.Fatalf("step %d: condition %d: want {%s,%s}, got {%s,%s}",
								i, j, want.Status, want.Reason, last.conditions[j].Status, last.conditions[j].Reason)
						}
					}
				}
				prevCalls = got
			}
		})
	}
}

func TestProbeStorageHealth(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, true)
	registerHealthMetrics()

	degraded := []storagev1.StorageHealthCondition{{
		Status: storagev1.StorageDegraded, Reason: "PoolFull", Message: "pool nearly full",
	}}

	type step struct {
		storageConditions []storagev1.StorageHealthCondition
		storageErr        error
		wantCallCount     int
	}

	tests := []struct {
		name                  string
		supportsStorageHealth bool
		steps                 []step
	}{
		{
			name:                  "bad health updates CSINode",
			supportsStorageHealth: true,
			steps: []step{
				{storageConditions: degraded, wantCallCount: 1},
			},
		},
		{
			name:                  "cleared health clears CSINode",
			supportsStorageHealth: true,
			steps: []step{
				{storageConditions: degraded, wantCallCount: 1},
				{storageConditions: nil, wantCallCount: 2},
			},
		},
		{
			// Manager always forwards; nodeinfomanager suppresses no-op UpdateStatus.
			name:                  "same condition twice still calls CSINode updater",
			supportsStorageHealth: true,
			steps: []step{
				{storageConditions: degraded, wantCallCount: 1},
				{storageConditions: degraded, wantCallCount: 2},
			},
		},
		{
			name:                  "error leaves previous conditions unchanged",
			supportsStorageHealth: true,
			steps: []step{
				{storageConditions: degraded, wantCallCount: 1},
				{storageErr: errors.New("rpc failed"), wantCallCount: 1},
			},
		},
		{
			name:                  "capability not supported skips probe",
			supportsStorageHealth: false,
			steps: []step{
				{wantCallCount: 0},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := &fakeHealthClient{supportsStorageHealth: tc.supportsStorageHealth}
			updater := &fakeCSINodeUpdater{}
			m := newTestManager(t, &testStatusUpdater{}, client)
			m.csiNodeUpdater = func() CSINodeUpdater { return updater }

			prevCalls := 0
			for i, s := range tc.steps {
				client.storageConditions = s.storageConditions
				client.storageErr = s.storageErr
				m.probeStorageHealth(context.Background())

				updater.mu.Lock()
				got := len(updater.calls)
				if got != s.wantCallCount {
					updater.mu.Unlock()
					t.Fatalf("step %d: expected %d CSINode update(s), got %d", i, s.wantCallCount, got)
				}
				if got > prevCalls {
					last := updater.calls[got-1]
					if last.driverName != "test.driver" {
						updater.mu.Unlock()
						t.Fatalf("step %d: unexpected driver: %s", i, last.driverName)
					}
					if len(last.conditions) != len(s.storageConditions) {
						updater.mu.Unlock()
						t.Fatalf("step %d: expected %d conditions, got %d", i, len(s.storageConditions), len(last.conditions))
					}
					for j, want := range s.storageConditions {
						if last.conditions[j].Status != want.Status || last.conditions[j].Reason != want.Reason {
							updater.mu.Unlock()
							t.Fatalf("step %d: condition %d: want {%s,%s}, got {%s,%s}",
								i, j, want.Status, want.Reason, last.conditions[j].Status, last.conditions[j].Reason)
						}
					}
				}
				updater.mu.Unlock()
				prevCalls = got
			}
		})
	}
}

func TestConditionsEqual(t *testing.T) {
	a := []v1.VolumeHealthCondition{{Status: v1.VolumeHealthDegraded, Reason: "A", Message: "old"}}
	b := []v1.VolumeHealthCondition{{Status: v1.VolumeHealthDegraded, Reason: "A", Message: "new"}}
	if !util.VolumeHealthConditionSetsEqual(a, b) {
		t.Fatal("message-only difference should be equal for (status,reason) set")
	}
	c := []v1.VolumeHealthCondition{{Status: v1.VolumeHealthDataLoss, Reason: "A"}}
	if util.VolumeHealthConditionSetsEqual(a, c) {
		t.Fatal("different status should not be equal")
	}
}
