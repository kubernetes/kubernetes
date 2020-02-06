// +build !providerless

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

package awsebs

import (
	"errors"
	"testing"

	"k8s.io/klog"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/legacy-cloud-providers/aws"
)

func TestGetVolumeName_Volume(t *testing.T) {
	plugin := newPlugin(t)
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
	plugin := newPlugin(t)
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
	attach attachCall
	detach detachCall
	t      *testing.T

	// Actual test to run
	test func(test *testcase) (string, error)
	// Expected return of the test
	expectedDevice string
	expectedError  error
}

func TestAttachDetach(t *testing.T) {
	diskName := aws.KubernetesVolumeID("disk")
	nodeName := types.NodeName("instance")
	spec := createVolSpec(diskName, false)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	tests := []testcase{
		// Successful Attach call
		{
			name:   "Attach_Positive",
			attach: attachCall{diskName, nodeName, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedDevice: "/dev/sda",
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
			name:   "Detach_Positive",
			detach: detachCall{diskName, nodeName, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				mountPath := "/mnt/" + string(diskName)
				return "", detacher.Detach(mountPath, nodeName)
			},
		},
		// Detach fails
		{
			name:   "Detach_Negative",
			detach: detachCall{diskName, nodeName, "", detachError},
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
func newPlugin(t *testing.T) *awsElasticBlockStorePlugin {
	host := volumetest.NewFakeVolumeHost(t, "/tmp", nil, nil)
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
		Volume: &v1.Volume{
			VolumeSource: v1.VolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: string(name),
					ReadOnly: readOnly,
				},
			},
		},
	}
}

func createPVSpec(name aws.KubernetesVolumeID, readOnly bool) *volume.Spec {
	return &volume.Spec{
		PersistentVolume: &v1.PersistentVolume{
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
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
	retDeviceName string
	ret           error
}

type detachCall struct {
	diskName      aws.KubernetesVolumeID
	nodeName      types.NodeName
	retDeviceName string
	ret           error
}

func (testcase *testcase) AttachDisk(diskName aws.KubernetesVolumeID, nodeName types.NodeName) (string, error) {
	expected := &testcase.attach

	if expected.diskName == "" && expected.nodeName == "" {
		// testcase.attach looks uninitialized, test did not expect to call
		// AttachDisk
		testcase.t.Errorf("Unexpected AttachDisk call!")
		return "", errors.New("unexpected AttachDisk call")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return "", errors.New("Unexpected AttachDisk call: wrong diskName")
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected AttachDisk call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return "", errors.New("Unexpected AttachDisk call: wrong nodeName")
	}

	klog.V(4).Infof("AttachDisk call: %s, %s, returning %q, %v", diskName, nodeName, expected.retDeviceName, expected.ret)

	return expected.retDeviceName, expected.ret
}

func (testcase *testcase) DetachDisk(diskName aws.KubernetesVolumeID, nodeName types.NodeName) (string, error) {
	expected := &testcase.detach

	if expected.diskName == "" && expected.nodeName == "" {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachDisk
		testcase.t.Errorf("Unexpected DetachDisk call!")
		return "", errors.New("unexpected DetachDisk call")
	}

	if expected.diskName != diskName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
		return "", errors.New("Unexpected DetachDisk call: wrong diskName")
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected DetachDisk call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return "", errors.New("Unexpected DetachDisk call: wrong nodeName")
	}

	klog.V(4).Infof("DetachDisk call: %s, %s, returning %q, %v", diskName, nodeName, expected.retDeviceName, expected.ret)

	return expected.retDeviceName, expected.ret
}

func (testcase *testcase) DiskIsAttached(diskName aws.KubernetesVolumeID, nodeName types.NodeName) (bool, error) {
	// DetachDisk no longer relies on DiskIsAttached api call
	return false, nil
}

func (testcase *testcase) DisksAreAttached(nodeDisks map[types.NodeName][]aws.KubernetesVolumeID) (map[types.NodeName]map[aws.KubernetesVolumeID]bool, error) {
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

func (testcase *testcase) ResizeDisk(
	volumeName aws.KubernetesVolumeID,
	oldSize resource.Quantity,
	newSize resource.Quantity) (resource.Quantity, error) {
	return oldSize, errors.New("Not implemented")
}
