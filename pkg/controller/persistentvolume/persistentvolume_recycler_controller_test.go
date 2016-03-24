/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package persistentvolume

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/host_path"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

const (
	mySyncPeriod   = 2 * time.Second
	myMaximumRetry = 3
)

func TestFailedRecycling(t *testing.T) {
	pv := preparePV()

	mockClient := &mockBinderClient{
		volume: pv,
	}

	// no Init called for pluginMgr and no plugins are available.  Volume should fail recycling.
	plugMgr := volume.VolumePluginMgr{}

	recycler := &PersistentVolumeRecycler{
		kubeClient:      fake.NewSimpleClientset(),
		client:          mockClient,
		pluginMgr:       plugMgr,
		releasedVolumes: make(map[string]releasedVolumeStatus),
	}

	err := recycler.reclaimVolume(pv)
	if err != nil {
		t.Errorf("Unexpected non-nil error: %v", err)
	}

	if mockClient.volume.Status.Phase != api.VolumeFailed {
		t.Errorf("Expected %s but got %s", api.VolumeFailed, mockClient.volume.Status.Phase)
	}

	// Use a new volume for the next test
	pv = preparePV()
	mockClient.volume = pv

	pv.Spec.PersistentVolumeReclaimPolicy = api.PersistentVolumeReclaimDelete
	err = recycler.reclaimVolume(pv)
	if err != nil {
		t.Errorf("Unexpected non-nil error: %v", err)
	}

	if mockClient.volume.Status.Phase != api.VolumeFailed {
		t.Errorf("Expected %s but got %s", api.VolumeFailed, mockClient.volume.Status.Phase)
	}
}

func TestRecyclingRetry(t *testing.T) {
	// Test that recycler controller retries to recycle a volume several times, which succeeds eventually
	pv := preparePV()

	mockClient := &mockBinderClient{
		volume: pv,
	}

	plugMgr := volume.VolumePluginMgr{}
	// Use a fake NewRecycler function
	plugMgr.InitPlugins(host_path.ProbeRecyclableVolumePlugins(newFailingMockRecycler, volume.VolumeConfig{}), volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil))
	// Reset a global call counter
	failedCallCount = 0

	recycler := &PersistentVolumeRecycler{
		kubeClient:      fake.NewSimpleClientset(),
		client:          mockClient,
		pluginMgr:       plugMgr,
		syncPeriod:      mySyncPeriod,
		maximumRetry:    myMaximumRetry,
		releasedVolumes: make(map[string]releasedVolumeStatus),
	}

	// All but the last attempt will fail
	testRecycleFailures(t, recycler, mockClient, pv, myMaximumRetry-1)

	// The last attempt should succeed
	err := recycler.reclaimVolume(pv)
	if err != nil {
		t.Errorf("Last step: Recycler failed: %v", err)
	}

	if mockClient.volume.Status.Phase != api.VolumePending {
		t.Errorf("Last step: The volume should be Pending, but is %s instead", mockClient.volume.Status.Phase)
	}
	// Check the cache, it should not have any entry
	status, found := recycler.releasedVolumes[pv.Name]
	if found {
		t.Errorf("Last step: Expected PV to be removed from cache, got %v", status)
	}
}

func TestRecyclingRetryAlwaysFail(t *testing.T) {
	// Test that recycler controller retries to recycle a volume several times, which always fails.
	pv := preparePV()

	mockClient := &mockBinderClient{
		volume: pv,
	}

	plugMgr := volume.VolumePluginMgr{}
	// Use a fake NewRecycler function
	plugMgr.InitPlugins(host_path.ProbeRecyclableVolumePlugins(newAlwaysFailingMockRecycler, volume.VolumeConfig{}), volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil))
	// Reset a global call counter
	failedCallCount = 0

	recycler := &PersistentVolumeRecycler{
		kubeClient:      fake.NewSimpleClientset(),
		client:          mockClient,
		pluginMgr:       plugMgr,
		syncPeriod:      mySyncPeriod,
		maximumRetry:    myMaximumRetry,
		releasedVolumes: make(map[string]releasedVolumeStatus),
	}

	// myMaximumRetry recycle attempts will fail
	testRecycleFailures(t, recycler, mockClient, pv, myMaximumRetry)

	// The volume should be failed after myMaximumRetry attempts
	err := recycler.reclaimVolume(pv)
	if err != nil {
		t.Errorf("Last step: Recycler failed: %v", err)
	}

	if mockClient.volume.Status.Phase != api.VolumeFailed {
		t.Errorf("Last step: The volume should be Failed, but is %s instead", mockClient.volume.Status.Phase)
	}
	// Check the cache, it should not have any entry
	status, found := recycler.releasedVolumes[pv.Name]
	if found {
		t.Errorf("Last step: Expected PV to be removed from cache, got %v", status)
	}
}

func preparePV() *api.PersistentVolume {
	return &api.PersistentVolume{
		Spec: api.PersistentVolumeSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("8Gi"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/tmp/data02",
				},
			},
			PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRecycle,
			ClaimRef: &api.ObjectReference{
				Name:      "foo",
				Namespace: "bar",
			},
		},
		Status: api.PersistentVolumeStatus{
			Phase: api.VolumeReleased,
		},
	}
}

// Test that `count` attempts to recycle a PV fails.
func testRecycleFailures(t *testing.T, recycler *PersistentVolumeRecycler, mockClient *mockBinderClient, pv *api.PersistentVolume, count int) {
	for i := 1; i <= count; i++ {
		err := recycler.reclaimVolume(pv)
		if err != nil {
			t.Errorf("STEP %d: Recycler faled: %v", i, err)
		}

		// Check the status, it should be failed
		if mockClient.volume.Status.Phase != api.VolumeReleased {
			t.Errorf("STEP %d: The volume should be Released, but is %s instead", i, mockClient.volume.Status.Phase)
		}

		// Check the failed volume cache
		status, found := recycler.releasedVolumes[pv.Name]
		if !found {
			t.Errorf("STEP %d: cannot find released volume status", i)
		}
		if status.retryCount != i {
			t.Errorf("STEP %d: Expected nr. of attempts to be %d, got %d", i, i, status.retryCount)
		}

		// call reclaimVolume too early, it should not increment the retryCount
		time.Sleep(mySyncPeriod / 2)
		err = recycler.reclaimVolume(pv)
		if err != nil {
			t.Errorf("STEP %d: Recycler failed: %v", i, err)
		}

		status, found = recycler.releasedVolumes[pv.Name]
		if !found {
			t.Errorf("STEP %d: cannot find released volume status", i)
		}
		if status.retryCount != i {
			t.Errorf("STEP %d: Expected nr. of attempts to be %d, got %d", i, i, status.retryCount)
		}

		// Call the next reclaimVolume() after full pvRecycleRetryPeriod
		time.Sleep(mySyncPeriod / 2)
	}
}

func newFailingMockRecycler(spec *volume.Spec, host volume.VolumeHost, config volume.VolumeConfig) (volume.Recycler, error) {
	return &failingMockRecycler{
		path:       spec.PersistentVolume.Spec.HostPath.Path,
		errorCount: myMaximumRetry - 1, // fail two times and then successfully recycle the volume
	}, nil
}

func newAlwaysFailingMockRecycler(spec *volume.Spec, host volume.VolumeHost, config volume.VolumeConfig) (volume.Recycler, error) {
	return &failingMockRecycler{
		path:       spec.PersistentVolume.Spec.HostPath.Path,
		errorCount: 1000, // always fail
	}, nil
}

type failingMockRecycler struct {
	path string
	// How many times should the recycler fail before returning success.
	errorCount int
	volume.MetricsNil
}

// Counter of failingMockRecycler.Recycle() calls. Global variable just for
// testing. It's too much code to create a custom volume plugin, which would
// hold this variable.
var failedCallCount = 0

func (r *failingMockRecycler) GetPath() string {
	return r.path
}

func (r *failingMockRecycler) Recycle() error {
	failedCallCount += 1
	if failedCallCount <= r.errorCount {
		return fmt.Errorf("Failing for %d. time", failedCallCount)
	}
	// return nil means recycle passed
	return nil
}
