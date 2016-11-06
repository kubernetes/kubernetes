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

package cinder

import (
	"errors"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/types"
)

func TestGetDeviceName_Volume(t *testing.T) {
	plugin := newPlugin()
	name := "my-cinder-volume"
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
	name := "my-cinder-pv"
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
	diskPath       diskPathCall
	t              *testing.T

	instanceID string
	// Actual test to run
	test func(test *testcase) (string, error)
	// Expected return of the test
	expectedDevice string
	expectedError  error
}

func TestAttachDetach(t *testing.T) {
	diskName := "disk"
	instanceID := "instance"
	nodeName := types.NodeName(instanceID)
	readOnly := false
	spec := createVolSpec(diskName, readOnly)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	diskCheckError := errors.New("Fake DiskIsAttached error")
	diskPathError := errors.New("Fake GetAttachmentDiskPath error")
	tests := []testcase{
		// Successful Attach call
		{
			name:           "Attach_Positive",
			instanceID:     instanceID,
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, nil},
			attach:         attachCall{diskName, instanceID, "", nil},
			diskPath:       diskPathCall{diskName, instanceID, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedDevice: "/dev/sda",
		},

		// Disk is already attached
		{
			name:           "Attach_Positive_AlreadyAttached",
			instanceID:     instanceID,
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, true, nil},
			diskPath:       diskPathCall{diskName, instanceID, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedDevice: "/dev/sda",
		},

		// DiskIsAttached fails and Attach succeeds
		{
			name:           "Attach_Positive_CheckFails",
			instanceID:     instanceID,
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, diskCheckError},
			attach:         attachCall{diskName, instanceID, "", nil},
			diskPath:       diskPathCall{diskName, instanceID, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedDevice: "/dev/sda",
		},

		// Attach call fails
		{
			name:           "Attach_Negative",
			instanceID:     instanceID,
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, diskCheckError},
			attach:         attachCall{diskName, instanceID, "/dev/sda", attachError},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedError: attachError,
		},

		// GetAttachmentDiskPath call fails
		{
			name:           "Attach_Negative_DiskPatchFails",
			instanceID:     instanceID,
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, diskCheckError},
			attach:         attachCall{diskName, instanceID, "", nil},
			diskPath:       diskPathCall{diskName, instanceID, "", diskPathError},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedError: diskPathError,
		},

		// Detach succeeds
		{
			name:           "Detach_Positive",
			instanceID:     instanceID,
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, true, nil},
			detach:         detachCall{diskName, instanceID, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, nodeName)
			},
		},

		// Disk is already detached
		{
			name:           "Detach_Positive_AlreadyDetached",
			instanceID:     instanceID,
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, nodeName)
			},
		},

		// Detach succeeds when DiskIsAttached fails
		{
			name:           "Detach_Positive_CheckFails",
			instanceID:     instanceID,
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, diskCheckError},
			detach:         detachCall{diskName, instanceID, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, nodeName)
			},
		},

		// Detach fails
		{
			name:           "Detach_Negative",
			instanceID:     instanceID,
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, diskCheckError},
			detach:         detachCall{diskName, instanceID, detachError},
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

// newPlugin creates a new gcePersistentDiskPlugin with fake cloud, NewAttacher
// and NewDetacher won't work.
func newPlugin() *cinderPlugin {
	host := volumetest.NewFakeVolumeHost("/tmp", nil, nil)
	plugins := ProbeVolumePlugins()
	plugin := plugins[0]
	plugin.Init(host)
	return plugin.(*cinderPlugin)
}

func newAttacher(testcase *testcase) *cinderDiskAttacher {
	return &cinderDiskAttacher{
		host:           nil,
		cinderProvider: testcase,
	}
}

func newDetacher(testcase *testcase) *cinderDiskDetacher {
	return &cinderDiskDetacher{
		cinderProvider: testcase,
	}
}

func createVolSpec(name string, readOnly bool) *volume.Spec {
	return &volume.Spec{
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				Cinder: &api.CinderVolumeSource{
					VolumeID: name,
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
					Cinder: &api.CinderVolumeSource{
						VolumeID: name,
						ReadOnly: readOnly,
					},
				},
			},
		},
	}
}

// Fake GCE implementation

type attachCall struct {
	diskName      string
	instanceID    string
	retDeviceName string
	ret           error
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

type diskPathCall struct {
	diskName, instanceID string
	retPath              string
	ret                  error
}

func (testcase *testcase) AttachDisk(instanceID string, diskName string) (string, error) {
	expected := &testcase.attach

	if expected.diskName == "" && expected.instanceID == "" {
		// testcase.attach looks uninitialized, test did not expect to call
		// AttachDisk
		testcase.t.Errorf("Unexpected AttachDisk call!")
		return "", errors.New("Unexpected AttachDisk call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return "", errors.New("Unexpected AttachDisk call: wrong diskName")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected AttachDisk call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return "", errors.New("Unexpected AttachDisk call: wrong instanceID")
	}

	glog.V(4).Infof("AttachDisk call: %s, %s, returning %q, %v", diskName, instanceID, expected.retDeviceName, expected.ret)

	return expected.retDeviceName, expected.ret
}

func (testcase *testcase) DetachDisk(instanceID string, partialDiskId string) error {
	expected := &testcase.detach

	if expected.devicePath == "" && expected.instanceID == "" {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachDisk
		testcase.t.Errorf("Unexpected DetachDisk call!")
		return errors.New("Unexpected DetachDisk call!")
	}

	if expected.devicePath != partialDiskId {
		testcase.t.Errorf("Unexpected DetachDisk call: expected partialDiskId %s, got %s", expected.devicePath, partialDiskId)
		return errors.New("Unexpected DetachDisk call: wrong diskName")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected DetachDisk call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return errors.New("Unexpected DetachDisk call: wrong instanceID")
	}

	glog.V(4).Infof("DetachDisk call: %s, %s, returning %v", partialDiskId, instanceID, expected.ret)

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

func (testcase *testcase) GetAttachmentDiskPath(instanceID string, diskName string) (string, error) {
	expected := &testcase.diskPath
	if expected.diskName == "" && expected.instanceID == "" {
		// testcase.diskPath looks uninitialized, test did not expect to
		// call GetAttachmentDiskPath
		testcase.t.Errorf("Unexpected GetAttachmentDiskPath call!")
		return "", errors.New("Unexpected GetAttachmentDiskPath call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected GetAttachmentDiskPath call: expected diskName %s, got %s", expected.diskName, diskName)
		return "", errors.New("Unexpected GetAttachmentDiskPath call: wrong diskName")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected GetAttachmentDiskPath call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return "", errors.New("Unexpected GetAttachmentDiskPath call: wrong instanceID")
	}

	glog.V(4).Infof("GetAttachmentDiskPath call: %s, %s, returning %v, %v", diskName, instanceID, expected.retPath, expected.ret)

	return expected.retPath, expected.ret
}

func (testcase *testcase) ShouldTrustDevicePath() bool {
	return true
}

func (testcase *testcase) CreateVolume(name string, size int, vtype, availability string, tags *map[string]string) (volumeName string, err error) {
	return "", errors.New("Not implemented")
}

func (testcase *testcase) GetDevicePath(diskId string) string {
	return ""
}

func (testcase *testcase) InstanceID() (string, error) {
	return testcase.instanceID, nil
}

func (testcase *testcase) DeleteVolume(volumeName string) error {
	return errors.New("Not implemented")
}

func (testcase *testcase) GetAutoLabelsForPD(name string) (map[string]string, error) {
	return map[string]string{}, errors.New("Not implemented")
}

func (testcase *testcase) Instances() (cloudprovider.Instances, bool) {
	return &instances{testcase.instanceID}, true
}

func (testcase *testcase) DisksAreAttached(diskNames []string, nodeName string) (map[string]bool, error) {
	return nil, errors.New("Not implemented")
}

// Implementation of fake cloudprovider.Instances
type instances struct {
	instanceID string
}

func (instances *instances) NodeAddresses(name types.NodeName) ([]api.NodeAddress, error) {
	return []api.NodeAddress{}, errors.New("Not implemented")
}

func (instances *instances) ExternalID(name types.NodeName) (string, error) {
	return "", errors.New("Not implemented")
}

func (instances *instances) InstanceID(name types.NodeName) (string, error) {
	return instances.instanceID, nil
}

func (instances *instances) InstanceType(name types.NodeName) (string, error) {
	return "", errors.New("Not implemented")
}

func (instances *instances) List(filter string) ([]types.NodeName, error) {
	return []types.NodeName{}, errors.New("Not implemented")
}

func (instances *instances) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("Not implemented")
}

func (instances *instances) CurrentNodeName(hostname string) (types.NodeName, error) {
	return "", errors.New("Not implemented")
}
