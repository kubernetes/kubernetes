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
	"k8s.io/kubernetes/pkg/types"
)

func TestGetVolumeName_Volume(t *testing.T) {
	plugin := newPlugin()
	name := aws.KubernetesVolumeID("my-aws-volume")
	spec := createVolSpec(name, false)

	volumeName, err := plugin.GetVolumeName(spec)
	if err != nil {
		t.Errorf("GetVolumeName error: %v", err)
	}
	if volumeName != string(name) {
		t.Errorf("GetVolumeName error: expected %s, got %s", name, volumeName)
	}
}

func TestGetVolumeName_PersistentVolume(t *testing.T) {
	plugin := newPlugin()
	name := aws.KubernetesVolumeID("my-aws-pv")
	spec := createPVSpec(name, true)

	volumeName, err := plugin.GetVolumeName(spec)
	if err != nil {
		t.Errorf("GetVolumeName error: %v", err)
	}
	if volumeName != string(name) {
		t.Errorf("GetVolumeName error: expected %s, got %s", name, volumeName)
	}
}

// One testcase for TestAttachDetach table test below
type testcase struct {
	name aws.KubernetesVolumeID
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
	diskName := aws.KubernetesVolumeID("disk")
	nodeName := types.NodeName("instance")
	readOnly := false
	spec := createVolSpec(diskName, readOnly)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	diskCheckError := errors.New("Fake DiskIsAttached error")
	tests := []testcase{
		// Successful Attach call
		{
			name:   "Attach_Positive",
			attach: attachCall{diskName, nodeName, readOnly, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedDevice: "/dev/sda",
		},

		// Attach call fails
		{
			name:   "Attach_Negative",
			attach: attachCall{diskName, nodeName, readOnly, "", attachError},
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
			detach:         detachCall{diskName, nodeName, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				mountPath := "/mnt/" + string(diskName)
				return "", detacher.Detach(mountPath, nodeName)
			},
		},

		// Disk is already detached
		{
			name:           "Detach_Positive_AlreadyDetached",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				mountPath := "/mnt/" + string(diskName)
				return "", detacher.Detach(mountPath, nodeName)
			},
		},

		// Detach succeeds when DiskIsAttached fails
		{
			name:           "Detach_Positive_CheckFails",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, diskCheckError},
			detach:         detachCall{diskName, nodeName, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				mountPath := "/mnt/" + string(diskName)
				return "", detacher.Detach(mountPath, nodeName)
			},
		},

		// Detach fails
		{
			name:           "Detach_Negative",
			diskIsAttached: diskIsAttachedCall{diskName, nodeName, false, diskCheckError},
			detach:         detachCall{diskName, nodeName, "", detachError},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				mountPath := "/mnt/" + string(diskName)
				return "", detacher.Detach(mountPath, nodeName)
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
	host := volumetest.NewFakeVolumeHost("/tmp", nil, nil)
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

func createVolSpec(name aws.KubernetesVolumeID, readOnly bool) *volume.Spec {
	return &volume.Spec{
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
					VolumeID: string(name),
					ReadOnly: readOnly,
				},
			},
		},
	}
}

func createPVSpec(name aws.KubernetesVolumeID, readOnly bool) *volume.Spec {
	return &volume.Spec{
		PersistentVolume: &api.PersistentVolume{
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
						VolumeID: string(name),
						ReadOnly: readOnly,
					},
				},
			},
		},
	}
}

// Fake AWS implementation

type attachCall struct {
	diskName      aws.KubernetesVolumeID
	nodeName      types.NodeName
	readOnly      bool
	retDeviceName string
	ret           error
}

type detachCall struct {
	diskName      aws.KubernetesVolumeID
	nodeName      types.NodeName
	retDeviceName string
	ret           error
}

type diskIsAttachedCall struct {
	diskName   aws.KubernetesVolumeID
	nodeName   types.NodeName
	isAttached bool
	ret        error
}

func (testcase *testcase) AttachDisk(diskName aws.KubernetesVolumeID, nodeName types.NodeName, readOnly bool) (string, error) {
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

	if expected.readOnly != readOnly {
		testcase.t.Errorf("Unexpected AttachDisk call: expected readOnly %v, got %v", expected.readOnly, readOnly)
		return "", errors.New("Unexpected AttachDisk call: wrong readOnly")
	}

	glog.V(4).Infof("AttachDisk call: %s, %s, %v, returning %q, %v", diskName, nodeName, readOnly, expected.retDeviceName, expected.ret)

	return expected.retDeviceName, expected.ret
}

func (testcase *testcase) DetachDisk(diskName aws.KubernetesVolumeID, nodeName types.NodeName) (string, error) {
	expected := &testcase.detach

	if expected.diskName == "" && expected.nodeName == "" {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachDisk
		testcase.t.Errorf("Unexpected DetachDisk call!")
		return "", errors.New("Unexpected DetachDisk call!")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return "", errors.New("Unexpected DetachDisk call: wrong diskName")
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return "", errors.New("Unexpected DetachDisk call: wrong nodeName")
	}

	glog.V(4).Infof("DetachDisk call: %s, %s, returning %q, %v", diskName, nodeName, expected.retDeviceName, expected.ret)

	return expected.retDeviceName, expected.ret
}

func (testcase *testcase) DiskIsAttached(diskName aws.KubernetesVolumeID, nodeName types.NodeName) (bool, error) {
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

func (testcase *testcase) DisksAreAttached(diskNames []aws.KubernetesVolumeID, nodeName types.NodeName) (map[aws.KubernetesVolumeID]bool, error) {
	return nil, errors.New("Not implemented")
}

func (testcase *testcase) CreateDisk(volumeOptions *aws.VolumeOptions) (volumeName aws.KubernetesVolumeID, err error) {
	return "", errors.New("Not implemented")
}

func (testcase *testcase) DeleteDisk(volumeName aws.KubernetesVolumeID) (bool, error) {
	return false, errors.New("Not implemented")
}

func (testcase *testcase) GetVolumeLabels(volumeName aws.KubernetesVolumeID) (map[string]string, error) {
	return map[string]string{}, errors.New("Not implemented")
}

func (testcase *testcase) GetDiskPath(volumeName aws.KubernetesVolumeID) (string, error) {
	return "", errors.New("Not implemented")
}
