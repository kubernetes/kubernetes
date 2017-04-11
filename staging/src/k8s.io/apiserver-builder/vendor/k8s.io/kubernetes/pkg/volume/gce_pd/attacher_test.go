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

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
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
	nodeName := types.NodeName("instance")
	readOnly := false
	spec := createVolSpec(diskName, readOnly)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	diskCheckError := errors.New("Fake DiskIsAttached error")
	tests := []testcase{
		// Successful Attach call
		{
			name:           "Attach_Positive",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, nil},
			attach:         attachCall{diskName, nodeName, readOnly, nil},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				devicePath, err := attacher.Attach(spec, nodeName)
				if devicePath != "/dev/disk/by-id/google-disk" {
					return fmt.Errorf("devicePath incorrect. Expected<\"/dev/disk/by-id/google-disk\"> Actual: <%q>", devicePath)
				}
				return err
			},
		},

		// Disk is already attached
		{
			name:           "Attach_Positive_AlreadyAttached",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, true, nil},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				devicePath, err := attacher.Attach(spec, nodeName)
				if devicePath != "/dev/disk/by-id/google-disk" {
					return fmt.Errorf("devicePath incorrect. Expected<\"/dev/disk/by-id/google-disk\"> Actual: <%q>", devicePath)
				}
				return err
			},
		},

		// DiskIsAttached fails and Attach succeeds
		{
			name:           "Attach_Positive_CheckFails",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, diskCheckError},
			attach:         attachCall{diskName, nodeName, readOnly, nil},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				devicePath, err := attacher.Attach(spec, nodeName)
				if devicePath != "/dev/disk/by-id/google-disk" {
					return fmt.Errorf("devicePath incorrect. Expected<\"/dev/disk/by-id/google-disk\"> Actual: <%q>", devicePath)
				}
				return err
			},
		},

		// Attach call fails
		{
			name:           "Attach_Negative",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, diskCheckError},
			attach:         attachCall{diskName, nodeName, readOnly, attachError},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				devicePath, err := attacher.Attach(spec, nodeName)
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
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, true, nil},
			detach:         detachCall{diskName, nodeName, nil},
			test: func(testcase *testcase) error {
				detacher := newDetacher(testcase)
				return detacher.Detach(diskName, nodeName)
			},
		},

		// Disk is already detached
		{
			name:           "Detach_Positive_AlreadyDetached",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, nil},
			test: func(testcase *testcase) error {
				detacher := newDetacher(testcase)
				return detacher.Detach(diskName, nodeName)
			},
		},

		// Detach succeeds when DiskIsAttached fails
		{
			name:           "Detach_Positive_CheckFails",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, diskCheckError},
			detach:         detachCall{diskName, nodeName, nil},
			test: func(testcase *testcase) error {
				detacher := newDetacher(testcase)
				return detacher.Detach(diskName, nodeName)
			},
		},

		// Detach fails
		{
			name:           "Detach_Negative",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, diskCheckError},
			detach:         detachCall{diskName, nodeName, detachError},
			test: func(testcase *testcase) error {
				detacher := newDetacher(testcase)
				return detacher.Detach(diskName, nodeName)
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
		nil /* plugins */)
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
		Volume: &v1.Volume{
			VolumeSource: v1.VolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName:   name,
					ReadOnly: readOnly,
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
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
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
	diskName string
	nodeName types.NodeName
	readOnly bool
	ret      error
}

type detachCall struct {
	devicePath string
	nodeName   types.NodeName
	ret        error
}

type diskIsAttachedCall struct {
	diskName   string
	nodeName   types.NodeName
	isAttached bool
	ret        error
}

func (testcase *testcase) AttachDisk(diskName string, nodeName types.NodeName, readOnly bool) error {
	expected := &testcase.attach

	if expected.diskName == "" && expected.nodeName == "" {
		// testcase.attach looks uninitialized, test did not expect to call
		// AttachDisk
		testcase.t.Errorf("Unexpected AttachDisk call!")
		return errors.New("Unexpected AttachDisk call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return errors.New("Unexpected AttachDisk call: wrong diskName")
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return errors.New("Unexpected AttachDisk call: wrong nodeName")
	}

	if expected.readOnly != readOnly {
		testcase.t.Errorf("Unexpected AttachDisk call: expected readOnly %v, got %v", expected.readOnly, readOnly)
		return errors.New("Unexpected AttachDisk call: wrong readOnly")
	}

	glog.V(4).Infof("AttachDisk call: %s, %s, %v, returning %v", diskName, nodeName, readOnly, expected.ret)

	return expected.ret
}

func (testcase *testcase) DetachDisk(devicePath string, nodeName types.NodeName) error {
	expected := &testcase.detach

	if expected.devicePath == "" && expected.nodeName == "" {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachDisk
		testcase.t.Errorf("Unexpected DetachDisk call!")
		return errors.New("Unexpected DetachDisk call!")
	}

	if expected.devicePath != devicePath {
		testcase.t.Errorf("Unexpected DetachDisk call: expected devicePath %s, got %s", expected.devicePath, devicePath)
		return errors.New("Unexpected DetachDisk call: wrong diskName")
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return errors.New("Unexpected DetachDisk call: wrong nodeName")
	}

	glog.V(4).Infof("DetachDisk call: %s, %s, returning %v", devicePath, nodeName, expected.ret)

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

func (testcase *testcase) DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error) {
	return nil, errors.New("Not implemented")
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
