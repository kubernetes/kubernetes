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

package gce_pd

import (
	"errors"
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	"github.com/golang/glog"
)

func TestGetDeviceName_Volume(t *testing.T) {
	plugin := newPlugin()
	name := "my-pd-volume"
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
	name := "my-pd-pv"
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
	// For fake GCE:
	attach         attachCall
	detach         detachCall
	diskIsAttached diskIsAttachedCall
	t              *testing.T

	// Actual test to run
	test func(test *testcase) error
	// Expected return of the test
	expectedReturn error
}

func TestAttachDetach(t *testing.T) {
	diskName := "disk"
	instanceID := "instance"
	readOnly := false
	spec := createVolSpec(diskName, readOnly)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	diskCheckError := errors.New("Fake DiskIsAttached error")
	tests := []testcase{
		// Successful Attach call
		{
			name:           "Attach_Positive",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, nil},
			attach:         attachCall{diskName, instanceID, readOnly, nil},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				devicePath, err := attacher.Attach(spec, instanceID)
				if devicePath != "/dev/disk/by-id/google-disk" {
					return fmt.Errorf("devicePath incorrect. Expected<\"/dev/disk/by-id/google-disk\"> Actual: <%q>", devicePath)
				}
				return err
			},
		},

		// Disk is already attached
		{
			name:           "Attach_Positive_AlreadyAttached",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, true, nil},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				devicePath, err := attacher.Attach(spec, instanceID)
				if devicePath != "/dev/disk/by-id/google-disk" {
					return fmt.Errorf("devicePath incorrect. Expected<\"/dev/disk/by-id/google-disk\"> Actual: <%q>", devicePath)
				}
				return err
			},
		},

		// DiskIsAttached fails and Attach succeeds
		{
			name:           "Attach_Positive_CheckFails",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, diskCheckError},
			attach:         attachCall{diskName, instanceID, readOnly, nil},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				devicePath, err := attacher.Attach(spec, instanceID)
				if devicePath != "/dev/disk/by-id/google-disk" {
					return fmt.Errorf("devicePath incorrect. Expected<\"/dev/disk/by-id/google-disk\"> Actual: <%q>", devicePath)
				}
				return err
			},
		},

		// Attach call fails
		{
			name:           "Attach_Negative",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, diskCheckError},
			attach:         attachCall{diskName, instanceID, readOnly, attachError},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				devicePath, err := attacher.Attach(spec, instanceID)
				if devicePath != "" {
					return fmt.Errorf("devicePath incorrect. Expected<\"\"> Actual: <%q>", devicePath)
				}
				return err
			},
			expectedReturn: attachError,
		},

		// Detach succeeds
		{
			name:           "Detach_Positive",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, true, nil},
			detach:         detachCall{diskName, instanceID, nil},
			test: func(testcase *testcase) error {
				detacher := newDetacher(testcase)
				return detacher.Detach(diskName, instanceID)
			},
		},

		// Disk is already detached
		{
			name:           "Detach_Positive_AlreadyDetached",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, nil},
			test: func(testcase *testcase) error {
				detacher := newDetacher(testcase)
				return detacher.Detach(diskName, instanceID)
			},
		},

		// Detach succeeds when DiskIsAttached fails
		{
			name:           "Detach_Positive_CheckFails",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, diskCheckError},
			detach:         detachCall{diskName, instanceID, nil},
			test: func(testcase *testcase) error {
				detacher := newDetacher(testcase)
				return detacher.Detach(diskName, instanceID)
			},
		},

		// Detach fails
		{
			name:           "Detach_Negative",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, diskCheckError},
			detach:         detachCall{diskName, instanceID, detachError},
			test: func(testcase *testcase) error {
				detacher := newDetacher(testcase)
				return detacher.Detach(diskName, instanceID)
			},
			expectedReturn: detachError,
		},
	}

	for _, testcase := range tests {
		testcase.t = t
		err := testcase.test(&testcase)
		if err != testcase.expectedReturn {
			t.Errorf("%s failed: expected err=%q, got %q", testcase.name, testcase.expectedReturn.Error(), err.Error())
		}
		t.Logf("Test %q succeeded", testcase.name)
	}
}

// newPlugin creates a new gcePersistentDiskPlugin with fake cloud, NewAttacher
// and NewDetacher won't work.
func newPlugin() *gcePersistentDiskPlugin {
	host := volumetest.NewFakeVolumeHost(
		"/tmp", /* rootDir */
		nil,    /* kubeClient */
		nil,    /* plugins */
		"" /* rootContext */)
	plugins := ProbeVolumePlugins()
	plugin := plugins[0]
	plugin.Init(host)
	return plugin.(*gcePersistentDiskPlugin)
}

func newAttacher(testcase *testcase) *gcePersistentDiskAttacher {
	return &gcePersistentDiskAttacher{
		host:     nil,
		gceDisks: testcase,
	}
}

func newDetacher(testcase *testcase) *gcePersistentDiskDetacher {
	return &gcePersistentDiskDetacher{
		gceDisks: testcase,
	}
}

func createVolSpec(name string, readOnly bool) *volume.Spec {
	return &volume.Spec{
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
					PDName:   name,
					ReadOnly: readOnly,
				},
			},
		},
	}
}

func createPVSpec(name string, readOnly bool) *volume.Spec {
	return &volume.Spec{
		PersistentVolume: &api.PersistentVolume{
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
						PDName:   name,
						ReadOnly: readOnly,
					},
				},
			},
		},
	}
}

// Fake GCE implementation

type attachCall struct {
	diskName   string
	instanceID string
	readOnly   bool
	ret        error
}

type detachCall struct {
	devicePath string
	instanceID string
	ret        error
}

type diskIsAttachedCall struct {
	diskName, instanceID string
	isAttached           bool
	ret                  error
}

func (testcase *testcase) AttachDisk(diskName, instanceID string, readOnly bool) error {
	expected := &testcase.attach

	if expected.diskName == "" && expected.instanceID == "" {
		// testcase.attach looks uninitialized, test did not expect to call
		// AttachDisk
		testcase.t.Errorf("Unexpected AttachDisk call!")
		return errors.New("Unexpected AttachDisk call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return errors.New("Unexpected AttachDisk call: wrong diskName")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected AttachDisk call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return errors.New("Unexpected AttachDisk call: wrong instanceID")
	}

	if expected.readOnly != readOnly {
		testcase.t.Errorf("Unexpected AttachDisk call: expected readOnly %v, got %v", expected.readOnly, readOnly)
		return errors.New("Unexpected AttachDisk call: wrong readOnly")
	}

	glog.V(4).Infof("AttachDisk call: %s, %s, %v, returning %v", diskName, instanceID, readOnly, expected.ret)

	return expected.ret
}

func (testcase *testcase) DetachDisk(devicePath, instanceID string) error {
	expected := &testcase.detach

	if expected.devicePath == "" && expected.instanceID == "" {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachDisk
		testcase.t.Errorf("Unexpected DetachDisk call!")
		return errors.New("Unexpected DetachDisk call!")
	}

	if expected.devicePath != devicePath {
		testcase.t.Errorf("Unexpected DetachDisk call: expected devicePath %s, got %s", expected.devicePath, devicePath)
		return errors.New("Unexpected DetachDisk call: wrong diskName")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected DetachDisk call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return errors.New("Unexpected DetachDisk call: wrong instanceID")
	}

	glog.V(4).Infof("DetachDisk call: %s, %s, returning %v", devicePath, instanceID, expected.ret)

	return expected.ret
}

func (testcase *testcase) DiskIsAttached(diskName, instanceID string) (bool, error) {
	expected := &testcase.diskIsAttached

	if expected.diskName == "" && expected.instanceID == "" {
		// testcase.diskIsAttached looks uninitialized, test did not expect to
		// call DiskIsAttached
		testcase.t.Errorf("Unexpected DiskIsAttached call!")
		return false, errors.New("Unexpected DiskIsAttached call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected DiskIsAttached call: expected diskName %s, got %s", expected.diskName, diskName)
		return false, errors.New("Unexpected DiskIsAttached call: wrong diskName")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected DiskIsAttached call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return false, errors.New("Unexpected DiskIsAttached call: wrong instanceID")
	}

	glog.V(4).Infof("DiskIsAttached call: %s, %s, returning %v, %v", diskName, instanceID, expected.isAttached, expected.ret)

	return expected.isAttached, expected.ret
}

func (testcase *testcase) CreateDisk(name string, diskType string, zone string, sizeGb int64, tags map[string]string) error {
	return errors.New("Not implemented")
}

func (testcase *testcase) DeleteDisk(diskToDelete string) error {
	return errors.New("Not implemented")
}

func (testcase *testcase) GetAutoLabelsForPD(name string, zone string) (map[string]string, error) {
	return map[string]string{}, errors.New("Not implemented")
}
