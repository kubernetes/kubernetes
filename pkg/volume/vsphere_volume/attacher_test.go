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

package vsphere_volume

import (
	"errors"
	"testing"

	"k8s.io/api/core/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
)

func TestGetDeviceName_Volume(t *testing.T) {
	plugin := newPlugin()
	volPath := "[local] volumes/test"
	spec := createVolSpec(volPath)

	deviceName, err := plugin.GetVolumeName(spec)
	if err != nil {
		t.Errorf("GetDeviceName error: %v", err)
	}
	if deviceName != volPath {
		t.Errorf("GetDeviceName error: expected %s, got %s", volPath, deviceName)
	}
}

func TestGetDeviceName_PersistentVolume(t *testing.T) {
	plugin := newPlugin()
	volPath := "[local] volumes/test"
	spec := createPVSpec(volPath)

	deviceName, err := plugin.GetVolumeName(spec)
	if err != nil {
		t.Errorf("GetDeviceName error: %v", err)
	}
	if deviceName != volPath {
		t.Errorf("GetDeviceName error: expected %s, got %s", volPath, deviceName)
	}
}

// One testcase for TestAttachDetach table test below
type testcase struct {
	name string
	// For fake vSphere:
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
	uuid := "00000000000000"
	diskName := "[local] volumes/test"
	nodeName := types.NodeName("host")
	spec := createVolSpec(diskName)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	diskCheckError := errors.New("Fake DiskIsAttached error")
	tests := []testcase{
		// Successful Attach call
		{
			name:   "Attach_Positive",
			attach: attachCall{diskName, nodeName, uuid, nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedDevice: "/dev/disk/by-id/wwn-0x" + uuid,
		},

		// Attach call fails
		{
			name:   "Attach_Negative",
			attach: attachCall{diskName, nodeName, "", attachError},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedError: attachError,
		},

		// Detach succeeds
		{
			name:           "Detach_Positive",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, true, nil},
			detach:         detachCall{diskName, nodeName, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, nodeName)
			},
		},

		// Disk is already detached
		{
			name:           "Detach_Positive_AlreadyDetached",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, nodeName)
			},
		},

		// Detach succeeds when DiskIsAttached fails
		{
			name:           "Detach_Positive_CheckFails",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, diskCheckError},
			detach:         detachCall{diskName, nodeName, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, nodeName)
			},
		},

		// Detach fails
		{
			name:           "Detach_Negative",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, diskCheckError},
			detach:         detachCall{diskName, nodeName, detachError},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, nodeName)
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

// newPlugin creates a new vsphereVolumePlugin with fake cloud, NewAttacher
// and NewDetacher won't work.
func newPlugin() *vsphereVolumePlugin {
	host := volumetest.NewFakeVolumeHost("/tmp", nil, nil)
	plugins := ProbeVolumePlugins()
	plugin := plugins[0]
	plugin.Init(host)
	return plugin.(*vsphereVolumePlugin)
}

func newAttacher(testcase *testcase) *vsphereVMDKAttacher {
	return &vsphereVMDKAttacher{
		host:           nil,
		vsphereVolumes: testcase,
	}
}

func newDetacher(testcase *testcase) *vsphereVMDKDetacher {
	return &vsphereVMDKDetacher{
		vsphereVolumes: testcase,
	}
}

func createVolSpec(name string) *volume.Spec {
	return &volume.Spec{
		Volume: &v1.Volume{
			VolumeSource: v1.VolumeSource{
				VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
					VolumePath: name,
				},
			},
		},
	}
}

func createPVSpec(name string) *volume.Spec {
	return &volume.Spec{
		PersistentVolume: &v1.PersistentVolume{
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
						VolumePath: name,
					},
				},
			},
		},
	}
}

// Fake vSphere implementation

type attachCall struct {
	diskName      string
	nodeName      types.NodeName
	retDeviceUUID string
	ret           error
}

type detachCall struct {
	diskName string
	nodeName types.NodeName
	ret      error
}

type diskIsAttachedCall struct {
	diskName   string
	nodeName   types.NodeName
	isAttached bool
	ret        error
}

func (testcase *testcase) AttachDisk(diskName string, storagePolicyName string, nodeName types.NodeName) (string, error) {
	expected := &testcase.attach

	if expected.diskName == "" && expected.nodeName == "" {
		// testcase.attach looks uninitialized, test did not expect to call
		// AttachDisk
		testcase.t.Errorf("Unexpected AttachDisk call!")
		return "", errors.New("Unexpected AttachDisk call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return "", errors.New("Unexpected AttachDisk call: wrong diskName")
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return "", errors.New("Unexpected AttachDisk call: wrong nodeName")
	}

	glog.V(4).Infof("AttachDisk call: %s, %s, returning %q, %v", diskName, nodeName, expected.retDeviceUUID, expected.ret)

	return expected.retDeviceUUID, expected.ret
}

func (testcase *testcase) DetachDisk(diskName string, nodeName types.NodeName) error {
	expected := &testcase.detach

	if expected.diskName == "" && expected.nodeName == "" {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachDisk
		testcase.t.Errorf("Unexpected DetachDisk call!")
		return errors.New("Unexpected DetachDisk call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return errors.New("Unexpected DetachDisk call: wrong diskName")
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return errors.New("Unexpected DetachDisk call: wrong nodeName")
	}

	glog.V(4).Infof("DetachDisk call: %s, %s, returning %v", diskName, nodeName, expected.ret)

	return expected.ret
}

func (testcase *testcase) DiskIsAttached(diskName string, nodeName types.NodeName) (bool, error) {
	expected := &testcase.diskIsAttached

	if expected.diskName == "" && expected.nodeName == "" {
		// testcase.diskIsAttached looks uninitialized, test did not expect to
		// call DiskIsAttached
		testcase.t.Errorf("Unexpected DiskIsAttached call!")
		return false, errors.New("Unexpected DiskIsAttached call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected DiskIsAttached call: expected diskName %s, got %s", expected.diskName, diskName)
		return false, errors.New("Unexpected DiskIsAttached call: wrong diskName")
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected DiskIsAttached call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return false, errors.New("Unexpected DiskIsAttached call: wrong nodeName")
	}

	glog.V(4).Infof("DiskIsAttached call: %s, %s, returning %v, %v", diskName, nodeName, expected.isAttached, expected.ret)

	return expected.isAttached, expected.ret
}

func (testcase *testcase) DisksAreAttached(nodeVolumes map[k8stypes.NodeName][]string) (map[k8stypes.NodeName]map[string]bool, error) {
	return nil, errors.New("Not implemented")
}

func (testcase *testcase) CreateVolume(volumeOptions *vclib.VolumeOptions) (volumePath string, err error) {
	return "", errors.New("Not implemented")
}

func (testcase *testcase) DeleteVolume(vmDiskPath string) error {
	return errors.New("Not implemented")
}
