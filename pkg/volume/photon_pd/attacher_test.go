/*
Copyright 2016 The Kubernetes Authors.

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

package photon_pd

import (
	"errors"
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/photon"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
)

func TestGetDeviceName_Volume(t *testing.T) {
	plugin := newPlugin()
	name := "my-photon-volume"
	spec := createVolSpec(name, false)

	deviceName, err := plugin.GetVolumeName(spec)
	if err != nil {
		t.Errorf("GetDeviceName error: %v", err)
	}
	if deviceName != name {
		t.Errorf("GetDeviceName error: expected %s, got %s", name, deviceName)
	}
}

func TestGetDeviceName_PersistentVolume(t *testing.T) {
	plugin := newPlugin()
	name := "my-photon-pv"
	spec := createPVSpec(name, true)

	deviceName, err := plugin.GetVolumeName(spec)
	if err != nil {
		t.Errorf("GetDeviceName error: %v", err)
	}
	if deviceName != name {
		t.Errorf("GetDeviceName error: expected %s, got %s", name, deviceName)
	}
}

// One testcase for TestAttachDetach table test below
type testcase struct {
	name string
	// For fake Photon cloud provider:
	attach         attachCall
	detach         detachCall
	diskIsAttached diskIsAttachedCall
	t              *testing.T

	// Actual test to run
	test func(test *testcase) (string, error)
	// Expected return of the test
	expectedDevice string
	expectedError  error
}

func TestAttachDetach(t *testing.T) {
	diskName := "000-000-000"
	node := types.NodeIdentifier{Name: types.NodeName("instance")}
	readOnly := false
	spec := createVolSpec(diskName, readOnly)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	diskCheckError := errors.New("Fake DiskIsAttached error")
	tests := []testcase{
		// Successful Attach call
		{
			name:   "Attach_Positive",
			attach: attachCall{diskName, node, nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, node)
			},
			expectedDevice: "/dev/disk/by-id/wwn-0x000000000",
		},

		// Attach call fails
		{
			name:   "Attach_Negative",
			attach: attachCall{diskName, node, attachError},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, node)
			},
			expectedError: attachError,
		},

		// Detach succeeds
		{
			name:           "Detach_Positive",
			diskIsAttached: diskIsAttachedCall{diskName, node, true, nil},
			detach:         detachCall{diskName, node, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, node)
			},
		},

		// Disk is already detached
		{
			name:           "Detach_Positive_AlreadyDetached",
			diskIsAttached: diskIsAttachedCall{diskName, node, false, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, node)
			},
		},

		// Detach succeeds when DiskIsAttached fails
		{
			name:           "Detach_Positive_CheckFails",
			diskIsAttached: diskIsAttachedCall{diskName, node, false, diskCheckError},
			detach:         detachCall{diskName, node, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, node)
			},
		},

		// Detach fails
		{
			name:           "Detach_Negative",
			diskIsAttached: diskIsAttachedCall{diskName, node, false, diskCheckError},
			detach:         detachCall{diskName, node, detachError},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, node)
			},
			expectedError: detachError,
		},
	}

	for _, testcase := range tests {
		testcase.t = t
		device, err := testcase.test(&testcase)
		if err != testcase.expectedError {
			t.Errorf("%s failed: expected err=%q, got %q", testcase.name, testcase.expectedError.Error(), err.Error())
		}
		if device != testcase.expectedDevice {
			t.Errorf("%s failed: expected device=%q, got %q", testcase.name, testcase.expectedDevice, device)
		}
		t.Logf("Test %q succeeded", testcase.name)
	}
}

// newPlugin creates a new gcePersistentDiskPlugin with fake cloud, NewAttacher
// and NewDetacher won't work.
func newPlugin() *photonPersistentDiskPlugin {
	host := volumetest.NewFakeVolumeHost("/tmp", nil, nil)
	plugins := ProbeVolumePlugins()
	plugin := plugins[0]
	plugin.Init(host)
	return plugin.(*photonPersistentDiskPlugin)
}

func newAttacher(testcase *testcase) *photonPersistentDiskAttacher {
	return &photonPersistentDiskAttacher{
		host:        nil,
		photonDisks: testcase,
	}
}

func newDetacher(testcase *testcase) *photonPersistentDiskDetacher {
	return &photonPersistentDiskDetacher{
		photonDisks: testcase,
	}
}

func createVolSpec(name string, readOnly bool) *volume.Spec {
	return &volume.Spec{
		Volume: &v1.Volume{
			VolumeSource: v1.VolumeSource{
				PhotonPersistentDisk: &v1.PhotonPersistentDiskVolumeSource{
					PdID: name,
				},
			},
		},
	}
}

func createPVSpec(name string, readOnly bool) *volume.Spec {
	return &volume.Spec{
		PersistentVolume: &v1.PersistentVolume{
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					PhotonPersistentDisk: &v1.PhotonPersistentDiskVolumeSource{
						PdID: name,
					},
				},
			},
		},
	}
}

// Fake PhotonPD implementation

type attachCall struct {
	diskName string
	node     types.NodeIdentifier
	ret      error
}

type detachCall struct {
	diskName string
	node     types.NodeIdentifier
	ret      error
}

type diskIsAttachedCall struct {
	diskName   string
	node       types.NodeIdentifier
	isAttached bool
	ret        error
}

func (testcase *testcase) AttachDisk(diskName string, nodeName types.NodeName) error {
	expected := &testcase.attach

	if expected.diskName == "" && expected.node.Name == "" {
		// testcase.attach looks uninitialized, test did not expect to call
		// AttachDisk
		testcase.t.Errorf("Unexpected AttachDisk call!")
		return errors.New("Unexpected AttachDisk call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return errors.New("Unexpected AttachDisk call: wrong diskName")
	}

	if expected.node.Name != nodeName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected nodeName %s, got %s", expected.node.Name, nodeName)
		return errors.New("Unexpected AttachDisk call: wrong nodeName")
	}

	glog.V(4).Infof("AttachDisk call: %s, %s, returning %v", diskName, nodeName, expected.ret)

	return expected.ret
}

func (testcase *testcase) DetachDisk(diskName string, nodeName types.NodeName) error {
	expected := &testcase.detach

	if expected.diskName == "" && expected.node.Name == "" {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachDisk
		testcase.t.Errorf("Unexpected DetachDisk call!")
		return errors.New("Unexpected DetachDisk call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return errors.New("Unexpected DetachDisk call: wrong diskName")
	}

	if expected.node.Name != nodeName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected nodeName %s, got %s", expected.node.Name, nodeName)
		return errors.New("Unexpected DetachDisk call: wrong nodeName")
	}

	glog.V(4).Infof("DetachDisk call: %s, %s, returning %v", diskName, nodeName, expected.ret)

	return expected.ret
}

func (testcase *testcase) DiskIsAttached(diskName string, nodeName types.NodeName) (bool, error) {
	expected := &testcase.diskIsAttached

	if expected.diskName == "" && expected.node.Name == "" {
		// testcase.diskIsAttached looks uninitialized, test did not expect to
		// call DiskIsAttached
		testcase.t.Errorf("Unexpected DiskIsAttached call!")
		return false, errors.New("Unexpected DiskIsAttached call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected DiskIsAttached call: expected diskName %s, got %s", expected.diskName, diskName)
		return false, errors.New("Unexpected DiskIsAttached call: wrong diskName")
	}

	if expected.node.Name != nodeName {
		testcase.t.Errorf("Unexpected DiskIsAttached call: expected nodeName %s, got %s", expected.node.Name, nodeName)
		return false, errors.New("Unexpected DiskIsAttached call: wrong nodeName")
	}

	glog.V(4).Infof("DiskIsAttached call: %s, %s, returning %v, %v", diskName, nodeName, expected.isAttached, expected.ret)

	return expected.isAttached, expected.ret
}

func (testcase *testcase) DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error) {
	return nil, errors.New("Not implemented")
}

func (testcase *testcase) CreateDisk(volumeOptions *photon.VolumeOptions) (volumeName string, err error) {
	return "", errors.New("Not implemented")
}

func (testcase *testcase) DeleteDisk(volumeName string) error {
	return errors.New("Not implemented")
}

func (testcase *testcase) GetVolumeLabels(volumeName string) (map[string]string, error) {
	return map[string]string{}, errors.New("Not implemented")
}

func (testcase *testcase) GetDiskPath(volumeName string) (string, error) {
	return "", errors.New("Not implemented")
}
