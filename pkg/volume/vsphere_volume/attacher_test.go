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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	"github.com/golang/glog"
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
	hostName := "host"
	spec := createVolSpec(diskName)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	diskCheckError := errors.New("Fake DiskIsAttached error")
	tests := []testcase{
		// Successful Attach call
		{
			name:   "Attach_Positive",
			attach: attachCall{diskName, hostName, uuid, nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, hostName)
			},
			expectedDevice: "/dev/disk/by-id/wwn-0x" + uuid,
		},

		// Attach call fails
		{
			name:   "Attach_Negative",
			attach: attachCall{diskName, hostName, "", attachError},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, hostName)
			},
			expectedError: attachError,
		},

		// Detach succeeds
		{
			name:           "Detach_Positive",
			diskIsAttached: diskIsAttachedCall{diskName, hostName, true, nil},
			detach:         detachCall{diskName, hostName, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, hostName)
			},
		},

		// Disk is already detached
		{
			name:           "Detach_Positive_AlreadyDetached",
			diskIsAttached: diskIsAttachedCall{diskName, hostName, false, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, hostName)
			},
		},

		// Detach succeeds when DiskIsAttached fails
		{
			name:           "Detach_Positive_CheckFails",
			diskIsAttached: diskIsAttachedCall{diskName, hostName, false, diskCheckError},
			detach:         detachCall{diskName, hostName, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, hostName)
			},
		},

		// Detach fails
		{
			name:           "Detach_Negative",
			diskIsAttached: diskIsAttachedCall{diskName, hostName, false, diskCheckError},
			detach:         detachCall{diskName, hostName, detachError},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, hostName)
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
	host := volumetest.NewFakeVolumeHost("/tmp", nil, nil, "")
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
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				VsphereVolume: &api.VsphereVirtualDiskVolumeSource{
					VolumePath: name,
				},
			},
		},
	}
}

func createPVSpec(name string) *volume.Spec {
	return &volume.Spec{
		PersistentVolume: &api.PersistentVolume{
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					VsphereVolume: &api.VsphereVirtualDiskVolumeSource{
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
	hostName      string
	retDeviceUUID string
	ret           error
}

type detachCall struct {
	diskName string
	hostName string
	ret      error
}

type diskIsAttachedCall struct {
	diskName, hostName string
	isAttached         bool
	ret                error
}

func (testcase *testcase) AttachDisk(diskName string, hostName string) (string, string, error) {
	expected := &testcase.attach

	if expected.diskName == "" && expected.hostName == "" {
		// testcase.attach looks uninitialized, test did not expect to call
		// AttachDisk
		testcase.t.Errorf("Unexpected AttachDisk call!")
		return "", "", errors.New("Unexpected AttachDisk call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return "", "", errors.New("Unexpected AttachDisk call: wrong diskName")
	}

	if expected.hostName != hostName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected hostName %s, got %s", expected.hostName, hostName)
		return "", "", errors.New("Unexpected AttachDisk call: wrong hostName")
	}

	glog.V(4).Infof("AttachDisk call: %s, %s, returning %q, %v", diskName, hostName, expected.retDeviceUUID, expected.ret)

	return "", expected.retDeviceUUID, expected.ret
}

func (testcase *testcase) DetachDisk(diskName string, hostName string) error {
	expected := &testcase.detach

	if expected.diskName == "" && expected.hostName == "" {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachDisk
		testcase.t.Errorf("Unexpected DetachDisk call!")
		return errors.New("Unexpected DetachDisk call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return errors.New("Unexpected DetachDisk call: wrong diskName")
	}

	if expected.hostName != hostName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected hostname %s, got %s", expected.hostName, hostName)
		return errors.New("Unexpected DetachDisk call: wrong hostname")
	}

	glog.V(4).Infof("DetachDisk call: %s, %s, returning %v", diskName, hostName, expected.ret)

	return expected.ret
}

func (testcase *testcase) DiskIsAttached(diskName, hostName string) (bool, error) {
	expected := &testcase.diskIsAttached

	if expected.diskName == "" && expected.hostName == "" {
		// testcase.diskIsAttached looks uninitialized, test did not expect to
		// call DiskIsAttached
		testcase.t.Errorf("Unexpected DiskIsAttached call!")
		return false, errors.New("Unexpected DiskIsAttached call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected DiskIsAttached call: expected diskName %s, got %s", expected.diskName, diskName)
		return false, errors.New("Unexpected DiskIsAttached call: wrong diskName")
	}

	if expected.hostName != hostName {
		testcase.t.Errorf("Unexpected DiskIsAttached call: expected hostName %s, got %s", expected.hostName, hostName)
		return false, errors.New("Unexpected DiskIsAttached call: wrong hostName")
	}

	glog.V(4).Infof("DiskIsAttached call: %s, %s, returning %v, %v", diskName, hostName, expected.isAttached, expected.ret)

	return expected.isAttached, expected.ret
}

func (testcase *testcase) CreateVolume(name string, size int, tags *map[string]string) (volumePath string, err error) {
	return "", errors.New("Not implemented")
}

func (testcase *testcase) DeleteVolume(vmDiskPath string) error {
	return errors.New("Not implemented")
}
