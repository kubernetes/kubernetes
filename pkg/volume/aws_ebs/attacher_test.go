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

package aws_ebs

import (
	"errors"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	"github.com/golang/glog"
)

func TestGetDeviceName_Volume(t *testing.T) {
	plugin := newPlugin()
	name := "my-aws-volume"
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
	name := "my-aws-pv"
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
	// For fake AWS:
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
			name:   "Attach_Positive",
			attach: attachCall{diskName, instanceID, readOnly, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, instanceID)
			},
			expectedDevice: "/dev/sda",
		},

		// Attach call fails
		{
			name:   "Attach_Negative",
			attach: attachCall{diskName, instanceID, readOnly, "", attachError},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, instanceID)
			},
			expectedError: attachError,
		},

		// Detach succeeds
		{
			name:           "Detach_Positive",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, true, nil},
			detach:         detachCall{diskName, instanceID, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, instanceID)
			},
		},

		// Disk is already detached
		{
			name:           "Detach_Positive_AlreadyDetached",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, instanceID)
			},
		},

		// Detach succeeds when DiskIsAttached fails
		{
			name:           "Detach_Positive_CheckFails",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, diskCheckError},
			detach:         detachCall{diskName, instanceID, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, instanceID)
			},
		},

		// Detach fails
		{
			name:           "Detach_Negative",
			diskIsAttached: diskIsAttachedCall{diskName, instanceID, false, diskCheckError},
			detach:         detachCall{diskName, instanceID, "", detachError},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(diskName, instanceID)
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
func newPlugin() *awsElasticBlockStorePlugin {
	host := volumetest.NewFakeVolumeHost("/tmp", nil, nil, "")
	plugins := ProbeVolumePlugins()
	plugin := plugins[0]
	plugin.Init(host)
	return plugin.(*awsElasticBlockStorePlugin)
}

func newAttacher(testcase *testcase) *awsElasticBlockStoreAttacher {
	return &awsElasticBlockStoreAttacher{
		host:       nil,
		awsVolumes: testcase,
	}
}

func newDetacher(testcase *testcase) *awsElasticBlockStoreDetacher {
	return &awsElasticBlockStoreDetacher{
		awsVolumes: testcase,
	}
}

func createVolSpec(name string, readOnly bool) *volume.Spec {
	return &volume.Spec{
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
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
					AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
						VolumeID: name,
						ReadOnly: readOnly,
					},
				},
			},
		},
	}
}

// Fake AWS implementation

type attachCall struct {
	diskName      string
	instanceID    string
	readOnly      bool
	retDeviceName string
	ret           error
}

type detachCall struct {
	diskName      string
	instanceID    string
	retDeviceName string
	ret           error
}

type diskIsAttachedCall struct {
	diskName, instanceID string
	isAttached           bool
	ret                  error
}

func (testcase *testcase) AttachDisk(diskName string, instanceID string, readOnly bool) (string, error) {
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

	if expected.readOnly != readOnly {
		testcase.t.Errorf("Unexpected AttachDisk call: expected readOnly %v, got %v", expected.readOnly, readOnly)
		return "", errors.New("Unexpected AttachDisk call: wrong readOnly")
	}

	glog.V(4).Infof("AttachDisk call: %s, %s, %v, returning %q, %v", diskName, instanceID, readOnly, expected.retDeviceName, expected.ret)

	return expected.retDeviceName, expected.ret
}

func (testcase *testcase) DetachDisk(diskName string, instanceID string) (string, error) {
	expected := &testcase.detach

	if expected.diskName == "" && expected.instanceID == "" {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachDisk
		testcase.t.Errorf("Unexpected DetachDisk call!")
		return "", errors.New("Unexpected DetachDisk call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return "", errors.New("Unexpected DetachDisk call: wrong diskName")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected DetachDisk call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return "", errors.New("Unexpected DetachDisk call: wrong instanceID")
	}

	glog.V(4).Infof("DetachDisk call: %s, %s, returning %q, %v", diskName, instanceID, expected.retDeviceName, expected.ret)

	return expected.retDeviceName, expected.ret
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

func (testcase *testcase) CreateDisk(volumeOptions *aws.VolumeOptions) (volumeName string, err error) {
	return "", errors.New("Not implemented")
}

func (testcase *testcase) DeleteDisk(volumeName string) (bool, error) {
	return false, errors.New("Not implemented")
}

func (testcase *testcase) GetVolumeLabels(volumeName string) (map[string]string, error) {
	return map[string]string{}, errors.New("Not implemented")
}

func (testcase *testcase) GetDiskPath(volumeName string) (string, error) {
	return "", errors.New("Not implemented")
}
