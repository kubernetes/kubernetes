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

package qingcloud_volume

import (
	"errors"
	"testing"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/qingcloud"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestGetDeviceName_Volume(t *testing.T) {
	plugin := newPlugin()
	name := "my-qingcloud-volume"
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
	name := "my-qingcloud-pv"
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

	attach           attachCall
	detach           detachCall
	volumeIsAttached VolumeIsAttachedCall
	t                *testing.T

	// Actual test to run
	test func(test *testcase) (string, error)
	// Expected return of the test
	expectedDevice string
	expectedError  error
}

func TestAttachDetach(t *testing.T) {
	volumeID := "test-qingcloud-volume-id"
	nodeName := types.NodeName("test-qingcloud-instance-name")
	readOnly := false
	spec := createVolSpec(volumeID, readOnly)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	volumeCheckError := errors.New("Fake VolumeIsAttached error")
	tests := []testcase{
		// Successful Attach call
		{
			name:   "Attach_Positive",
			attach: attachCall{volumeID, nodeName, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedDevice: "/dev/sda",
		},

		// Attach call fails
		{
			name:   "Attach_Negative",
			attach: attachCall{volumeID, nodeName, "", attachError},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedError: attachError,
		},

		// Detach succeeds
		{
			name:             "Detach_Positive",
			volumeIsAttached: VolumeIsAttachedCall{volumeID, nodeName, true, nil},
			detach:           detachCall{volumeID, nodeName, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(volumeID, nodeName)
			},
		},

		// Volume is already detached
		{
			name:             "Detach_Positive_AlreadyDetached",
			volumeIsAttached: VolumeIsAttachedCall{volumeID, nodeName, false, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(volumeID, nodeName)
			},
		},

		// Detach succeeds when VolumeIsAttached fails
		{
			name:             "Detach_Positive_CheckFails",
			volumeIsAttached: VolumeIsAttachedCall{volumeID, nodeName, false, volumeCheckError},
			detach:           detachCall{volumeID, nodeName, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(volumeID, nodeName)
			},
		},

		// Detach fails
		{
			name:             "Detach_Negative",
			volumeIsAttached: VolumeIsAttachedCall{volumeID, nodeName, false, volumeCheckError},
			detach:           detachCall{volumeID, nodeName, detachError},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(volumeID, nodeName)
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

// newPlugin creates a new qingcloudVolumePlugin with fake cloud, NewAttacher
// and NewDetacher won't work.
func newPlugin() *qingcloudVolumePlugin {
	host := volumetest.NewFakeVolumeHost("/tmp", nil, nil, "")
	plugins := ProbeVolumePlugins()
	plugin := plugins[0]
	plugin.Init(host)
	return plugin.(*qingcloudVolumePlugin)
}

func newAttacher(testcase *testcase) *qingcloudVolumeAttacher {
	return &qingcloudVolumeAttacher{
		host:      nil,
		qcVolumes: testcase,
	}
}

func newDetacher(testcase *testcase) *qingcloudVolumeDetacher {
	return &qingcloudVolumeDetacher{
		qingVolumes: testcase,
	}
}

func createVolSpec(name string, readOnly bool) *volume.Spec {
	return &volume.Spec{
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				QingCloudStore: &api.QingCloudStoreVolumeSource{
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
					QingCloudStore: &api.QingCloudStoreVolumeSource{
						VolumeID: name,
						ReadOnly: readOnly,
					},
				},
			},
		},
	}
}

// Fake qingcloud implementation

type attachCall struct {
	volumeID      string
	nodeName      types.NodeName
	retDeviceName string
	ret           error
}

type detachCall struct {
	volumeID string
	nodeName types.NodeName
	ret      error
}

type VolumeIsAttachedCall struct {
	volumeID   string
	nodeName   types.NodeName
	isAttached bool
	ret        error
}

func (testcase *testcase) AttachVolume(volumeID string, nodeName types.NodeName) (string, error) {
	expected := &testcase.attach

	if expected.volumeID == "" && expected.nodeName == "" {
		// testcase.attach looks uninitialized, test did not expect to call
		// AttachVolume
		testcase.t.Errorf("Unexpected AttachVolume call!")
		return "", errors.New("Unexpected AttachVolume call!")
	}

	if expected.volumeID != volumeID {
		testcase.t.Errorf("Unexpected AttachVolume call: expected volumeID %s, got %s", expected.volumeID, volumeID)
		return "", errors.New("Unexpected AttachVolume call: wrong volumeID")
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected AttachVolume call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return "", errors.New("Unexpected AttachVolume call: wrong nodeName")
	}

	glog.V(4).Infof("AttachVolume call: %s, %s, returning %q, %v", volumeID, nodeName, expected.retDeviceName, expected.ret)

	return expected.retDeviceName, expected.ret
}

func (testcase *testcase) DetachVolume(volumeID string, nodeName types.NodeName) error {
	expected := &testcase.detach

	if expected.volumeID == "" && expected.nodeName == "" {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachVolume
		testcase.t.Errorf("Unexpected DetachVolume call!")
		return errors.New("Unexpected DetachVolume call!")
	}

	if expected.volumeID != volumeID {
		testcase.t.Errorf("Unexpected DetachVolume call: expected volumeID %s, got %s", expected.volumeID, volumeID)
		return errors.New("Unexpected DetachVolume call: wrong volumeID")
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected DetachVolume call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return errors.New("Unexpected DetachVolume call: wrong nodeName")
	}

	glog.V(4).Infof("DetachVolume call: %s, %s, returning %v", volumeID, nodeName, expected.ret)

	return expected.ret
}

func (testcase *testcase) VolumeIsAttached(volumeID string, nodeName types.NodeName) (bool, error) {
	expected := &testcase.volumeIsAttached

	if expected.volumeID == "" && expected.nodeName == "" {
		// testcase.VolumeIsAttached looks uninitialized, test did not expect to
		// call VolumeIsAttached
		testcase.t.Errorf("Unexpected VolumeIsAttached call!")
		return false, errors.New("Unexpected VolumeIsAttached call!")
	}

	if expected.volumeID != volumeID {
		testcase.t.Errorf("Unexpected VolumeIsAttached call: expected volumeID %s, got %s", expected.volumeID, volumeID)
		return false, errors.New("Unexpected VolumeIsAttached call: wrong volumeID")
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected VolumeIsAttached call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return false, errors.New("Unexpected VolumeIsAttached call: wrong nodeName")
	}

	glog.V(4).Infof("VolumeIsAttached call: %s, %s, returning %v, %v", volumeID, nodeName, expected.isAttached, expected.ret)

	return expected.isAttached, expected.ret
}

func (testcase *testcase) CreateVolume(volumeOptions *qingcloud.VolumeOptions) (volumeID string, err error) {
	return "", errors.New("Not implemented")
}

func (testcase *testcase) DeleteVolume(volumeID string) (bool, error) {
	return false, errors.New("Not implemented")
}
