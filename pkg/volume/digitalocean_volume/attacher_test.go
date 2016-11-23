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

package digitalocean_volume

import (
	"errors"
	"strconv"
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
	name := "my-digitalocean-volume"
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
	name := "my-digitalocean-pv"
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
	// For fake DigitalOcean:
	attach         attachCall
	detach         detachCall
	diskIsAttached diskIsAttachedCall
	diskPath       diskPathCall
	t              *testing.T

	instanceID int
	// Actual test to run
	test func(test *testcase) (string, error)
	// Expected return of the test
	expectedDevice string
	expectedError  error
}

func TestAttachDetach(t *testing.T) {
	diskName := "disk"
	var instanceID int = 1234567
	nodeName := types.NodeName(strconv.Itoa(instanceID))
	readOnly := false
	spec := createVolSpec(diskName, readOnly)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	diskCheckError := errors.New("Fake VolumeIsAttached error")
	diskPathError := errors.New("Fake GetAttachmentVolumePath error")
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

		// Volume is already attached
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

		// VolumeIsAttached fails and Attach succeeds
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

		// GetAttachmentVolumePath call fails
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

		// Volume is already detached
		{
			name:           "Detach_Positive_AlreadyDetached",
			instanceID:     instanceID,
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, nodeName)
			},
		},

		// Detach succeeds when VolumeIsAttached fails
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
		t.Logf("Test %q started", testcase.name)
		device, err := testcase.test(&testcase)
		if err != testcase.expectedError {
			t.Errorf("%s failed: expected err=-, got %q", testcase.name, err.Error())
			//t.Errorf("%s failed: expected err=%q, got %q", testcase.name, testcase.expectedError.Error(), err.Error())
		}
		if device != testcase.expectedDevice {
			t.Errorf("%s failed: expected device=%q, got %q", testcase.name, testcase.expectedDevice, device)
		}
		t.Logf("Test %q succeeded", testcase.name)
	}
}

// newPlugin creates a new gcePersistentDiskPlugin with fake cloud, NewAttacher
// and NewDetacher won't work.
func newPlugin() *doVolumePlugin {
	host := volumetest.NewFakeVolumeHost("/tmp", nil, nil, "")
	plugins := ProbeVolumePlugins()
	plugin := plugins[0]
	plugin.Init(host)
	return plugin.(*doVolumePlugin)
}

func newAttacher(testcase *testcase) *doVolumeAttacher {
	return &doVolumeAttacher{
		host:       nil,
		doProvider: testcase,
	}
}

func newDetacher(testcase *testcase) *doVolumeDetacher {
	return &doVolumeDetacher{
		doProvider: testcase,
	}
}

func createVolSpec(name string, readOnly bool) *volume.Spec {
	return &volume.Spec{
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				DigitalOceanVolume: &api.DigitalOceanVolumeSource{
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
					DigitalOceanVolume: &api.DigitalOceanVolumeSource{
						VolumeID: name,
						ReadOnly: readOnly,
					},
				},
			},
		},
	}
}

// Fake DigitalOcean implementation

type attachCall struct {
	diskName      string
	instanceID    int
	retDeviceName string
	ret           error
}

type detachCall struct {
	devicePath string
	instanceID int
	ret        error
}

type diskIsAttachedCall struct {
	diskName   string
	instanceID int
	isAttached bool
	ret        error
}

type diskPathCall struct {
	diskName   string
	instanceID int
	retPath    string
	ret        error
}

func (testcase *testcase) AttachVolume(instanceID int, diskName string) (string, error) {
	expected := &testcase.attach

	if expected.diskName == "" && expected.instanceID == 0 {
		// testcase.attach looks uninitialized, test did not expect to call
		// AttachVolume
		testcase.t.Errorf("Unexpected AttachVolume call!")
		return "", errors.New("Unexpected AttachVolume call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected AttachVolume call: expected diskName %s, got %s", expected.diskName, diskName)
		return "", errors.New("Unexpected AttachVolume call: wrong diskName")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected AttachVolume call: expected instanceID %d, got %d", expected.instanceID, instanceID)
		return "", errors.New("Unexpected AttachVolume call: wrong instanceID")
	}

	glog.V(4).Infof("AttachVolume call: %s, %d, returning %q, %v", diskName, instanceID, expected.retDeviceName, expected.ret)

	return expected.retDeviceName, expected.ret
}

func (testcase *testcase) DetachVolume(instanceID int, partialDiskId string) error {
	expected := &testcase.detach

	if expected.devicePath == "" && expected.instanceID == 0 {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachVolume
		testcase.t.Errorf("Unexpected DetachVolume call!")
		return errors.New("Unexpected DetachVolume call!")
	}

	if expected.devicePath != partialDiskId {
		testcase.t.Errorf("Unexpected DetachVolume call: expected partialDiskId %s, got %s", expected.devicePath, partialDiskId)
		return errors.New("Unexpected DetachVolume call: wrong diskName")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected DetachVolume call: expected instanceID %s, got %d", expected.instanceID, instanceID)
		return errors.New("Unexpected DetachVolume call: wrong instanceID")
	}

	glog.V(4).Infof("DetachVolume call: %s, %d, returning %v", partialDiskId, instanceID, expected.ret)

	return expected.ret
}

func (testcase *testcase) VolumeIsAttached(diskName string, instanceID int) (bool, error) {
	expected := &testcase.diskIsAttached

	if expected.diskName == "" && expected.instanceID == 0 {
		// testcase.diskIsAttached looks uninitialized, test did not expect to
		// call VolumeIsAttached
		testcase.t.Errorf("Unexpected VolumeIsAttached call!")
		return false, errors.New("Unexpected VolumeIsAttached call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected VolumeIsAttached call: expected diskName %s, got %s", expected.diskName, diskName)
		return false, errors.New("Unexpected VolumeIsAttached call: wrong diskName")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected VolumeIsAttached call: expected instanceID %d, got %d", expected.instanceID, instanceID)
		return false, errors.New("Unexpected VolumeIsAttached call: wrong instanceID")
	}

	glog.V(4).Infof("VolumeIsAttached call: %s, %s, returning %v, %v", diskName, instanceID, expected.isAttached, expected.ret)

	return expected.isAttached, expected.ret
}

func (testcase *testcase) GetAttachmentVolumePath(instanceID int, diskName string) (string, error) {
	expected := &testcase.diskPath
	if expected.diskName == "" && expected.instanceID == 0 {
		// testcase.diskPath looks uninitialized, test did not expect to
		// call GetAttachmentVolumePath
		testcase.t.Errorf("Unexpected GetAttachmentVolumePath call!")
		return "", errors.New("Unexpected GetAttachmentVolumePath call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected GetAttachmentVolumePath call: expected diskName %s, got %s", expected.diskName, diskName)
		return "", errors.New("Unexpected GetAttachmentVolumePath call: wrong diskName")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected GetAttachmentVolumePath call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return "", errors.New("Unexpected GetAttachmentVolumePath call: wrong instanceID")
	}

	glog.V(4).Infof("GetAttachmentVolumePath call: %s, %s, returning %v, %v", diskName, instanceID, expected.retPath, expected.ret)

	return expected.retPath, expected.ret
}

func (testcase *testcase) CreateVolume(region string, name string, description string, sizeGigaBytes int64) (volumeName string, err error) {
	return "", errors.New("Not implemented")
}

func (testcase *testcase) GetDevicePath(diskId string) string {
	return ""
}

func (testcase *testcase) InstanceID() (string, error) {
	return strconv.Itoa(testcase.instanceID), nil
}
func (testcase *testcase) LocalInstanceID() (string, error) {
	return strconv.Itoa(testcase.instanceID), nil
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

func (testcase *testcase) VolumesAreAttached(diskNames []string, nodeName int) (map[string]bool, error) {
	return nil, errors.New("Not implemented")
}

func (testcase *testcase) GetRegion() string {
	return ""
}

// Implementation of fake cloudprovider.Instances
type instances struct {
	instanceID int
}

func (instances *instances) NodeAddresses(name types.NodeName) ([]api.NodeAddress, error) {
	return []api.NodeAddress{}, errors.New("Not implemented")
}

func (instances *instances) ExternalID(name types.NodeName) (string, error) {
	return "", errors.New("Not implemented")
}

func (instances *instances) InstanceID(name types.NodeName) (string, error) {
	return strconv.Itoa(instances.instanceID), nil
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
