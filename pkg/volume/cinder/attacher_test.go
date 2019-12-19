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

package cinder

import (
	"context"
	"errors"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	"fmt"
	"sort"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"
)

const (
	VolumeStatusPending = "pending"
	VolumeStatusDone    = "done"
)

var attachStatus = "Attach"
var detachStatus = "Detach"

func TestGetDeviceName_Volume(t *testing.T) {
	plugin := newPlugin(t)
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
	plugin := newPlugin(t)
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

func TestGetDeviceMountPath(t *testing.T) {
	name := "cinder-volume-id"
	spec := createVolSpec(name, false)
	rootDir := "/var/lib/kubelet/"
	host := volumetest.NewFakeVolumeHost(t, rootDir, nil, nil)

	attacher := &cinderDiskAttacher{
		host: host,
	}

	//test the path
	path, err := attacher.GetDeviceMountPath(spec)
	if err != nil {
		t.Errorf("Get device mount path error")
	}
	expectedPath := rootDir + "plugins/kubernetes.io/cinder/mounts/" + name
	if path != expectedPath {
		t.Errorf("Device mount path error: expected %s, got %s ", expectedPath, path)
	}
}

// One testcase for TestAttachDetach table test below
type testcase struct {
	name string
	// For fake GCE:
	attach           attachCall
	detach           detachCall
	operationPending operationPendingCall
	diskIsAttached   diskIsAttachedCall
	disksAreAttached disksAreAttachedCall
	diskPath         diskPathCall
	t                *testing.T
	attachOrDetach   *string

	instanceID string
	// Actual test to run
	test func(test *testcase) (string, error)
	// Expected return of the test
	expectedResult string
	expectedError  error
}

func TestAttachDetach(t *testing.T) {
	volumeID := "disk"
	instanceID := "instance"
	pending := VolumeStatusPending
	done := VolumeStatusDone
	nodeName := types.NodeName("nodeName")
	readOnly := false
	spec := createVolSpec(volumeID, readOnly)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	diskCheckError := errors.New("Fake DiskIsAttached error")
	diskPathError := errors.New("Fake GetAttachmentDiskPath error")
	disksCheckError := errors.New("Fake DisksAreAttached error")
	operationFinishTimeout := errors.New("Fake waitOperationFinished error")
	tests := []testcase{
		// Successful Attach call
		{
			name:             "Attach_Positive",
			instanceID:       instanceID,
			operationPending: operationPendingCall{volumeID, false, done, nil},
			diskIsAttached:   diskIsAttachedCall{instanceID, nodeName, volumeID, false, nil},
			attach:           attachCall{instanceID, volumeID, "", nil},
			diskPath:         diskPathCall{instanceID, volumeID, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedResult: "/dev/sda",
		},

		// Disk is already attached
		{
			name:             "Attach_Positive_AlreadyAttached",
			instanceID:       instanceID,
			operationPending: operationPendingCall{volumeID, false, done, nil},
			diskIsAttached:   diskIsAttachedCall{instanceID, nodeName, volumeID, true, nil},
			diskPath:         diskPathCall{instanceID, volumeID, "/dev/sda", nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedResult: "/dev/sda",
		},

		// Disk is attaching
		{
			name:             "Attach_is_attaching",
			instanceID:       instanceID,
			operationPending: operationPendingCall{volumeID, true, pending, operationFinishTimeout},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedError: operationFinishTimeout,
		},

		// Attach call fails
		{
			name:             "Attach_Negative",
			instanceID:       instanceID,
			operationPending: operationPendingCall{volumeID, false, done, nil},
			diskIsAttached:   diskIsAttachedCall{instanceID, nodeName, volumeID, false, diskCheckError},
			attach:           attachCall{instanceID, volumeID, "/dev/sda", attachError},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedError: attachError,
		},

		// GetAttachmentDiskPath call fails
		{
			name:             "Attach_Negative_DiskPatchFails",
			instanceID:       instanceID,
			operationPending: operationPendingCall{volumeID, false, done, nil},
			diskIsAttached:   diskIsAttachedCall{instanceID, nodeName, volumeID, false, nil},
			attach:           attachCall{instanceID, volumeID, "", nil},
			diskPath:         diskPathCall{instanceID, volumeID, "", diskPathError},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				return attacher.Attach(spec, nodeName)
			},
			expectedError: diskPathError,
		},

		// Successful VolumesAreAttached call, attached
		{
			name:             "VolumesAreAttached_Positive",
			instanceID:       instanceID,
			disksAreAttached: disksAreAttachedCall{instanceID, nodeName, []string{volumeID}, map[string]bool{volumeID: true}, nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				attachments, err := attacher.VolumesAreAttached([]*volume.Spec{spec}, nodeName)
				return serializeAttachments(attachments), err
			},
			expectedResult: serializeAttachments(map[*volume.Spec]bool{spec: true}),
		},

		// Successful VolumesAreAttached call, not attached
		{
			name:             "VolumesAreAttached_Negative",
			instanceID:       instanceID,
			disksAreAttached: disksAreAttachedCall{instanceID, nodeName, []string{volumeID}, map[string]bool{volumeID: false}, nil},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				attachments, err := attacher.VolumesAreAttached([]*volume.Spec{spec}, nodeName)
				return serializeAttachments(attachments), err
			},
			expectedResult: serializeAttachments(map[*volume.Spec]bool{spec: false}),
		},

		// Treat as attached when DisksAreAttached call fails
		{
			name:             "VolumesAreAttached_CinderFailed",
			instanceID:       instanceID,
			disksAreAttached: disksAreAttachedCall{instanceID, nodeName, []string{volumeID}, nil, disksCheckError},
			test: func(testcase *testcase) (string, error) {
				attacher := newAttacher(testcase)
				attachments, err := attacher.VolumesAreAttached([]*volume.Spec{spec}, nodeName)
				return serializeAttachments(attachments), err
			},
			expectedResult: serializeAttachments(map[*volume.Spec]bool{spec: true}),
			expectedError:  disksCheckError,
		},

		// Detach succeeds
		{
			name:             "Detach_Positive",
			instanceID:       instanceID,
			operationPending: operationPendingCall{volumeID, false, done, nil},
			diskIsAttached:   diskIsAttachedCall{instanceID, nodeName, volumeID, true, nil},
			detach:           detachCall{instanceID, volumeID, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(volumeID, nodeName)
			},
		},

		// Disk is already detached
		{
			name:             "Detach_Positive_AlreadyDetached",
			instanceID:       instanceID,
			operationPending: operationPendingCall{volumeID, false, done, nil},
			diskIsAttached:   diskIsAttachedCall{instanceID, nodeName, volumeID, false, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(volumeID, nodeName)
			},
		},

		// Detach succeeds when DiskIsAttached fails
		{
			name:             "Detach_Positive_CheckFails",
			instanceID:       instanceID,
			operationPending: operationPendingCall{volumeID, false, done, nil},
			diskIsAttached:   diskIsAttachedCall{instanceID, nodeName, volumeID, false, diskCheckError},
			detach:           detachCall{instanceID, volumeID, nil},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(volumeID, nodeName)
			},
		},

		// Detach fails
		{
			name:             "Detach_Negative",
			instanceID:       instanceID,
			operationPending: operationPendingCall{volumeID, false, done, nil},
			diskIsAttached:   diskIsAttachedCall{instanceID, nodeName, volumeID, false, diskCheckError},
			detach:           detachCall{instanceID, volumeID, detachError},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(volumeID, nodeName)
			},
			expectedError: detachError,
		},

		// // Disk is detaching
		{
			name:             "Detach_Is_Detaching",
			instanceID:       instanceID,
			operationPending: operationPendingCall{volumeID, true, pending, operationFinishTimeout},
			test: func(testcase *testcase) (string, error) {
				detacher := newDetacher(testcase)
				return "", detacher.Detach(volumeID, nodeName)
			},
			expectedError: operationFinishTimeout,
		},
	}

	for _, testcase := range tests {
		testcase.t = t
		attachOrDetach := ""
		testcase.attachOrDetach = &attachOrDetach
		result, err := testcase.test(&testcase)
		if err != testcase.expectedError {
			t.Errorf("%s failed: expected err=%q, got %q", testcase.name, testcase.expectedError, err)
		}
		if result != testcase.expectedResult {
			t.Errorf("%s failed: expected result=%q, got %q", testcase.name, testcase.expectedResult, result)
		}
	}
}

type volumeAttachmentFlag struct {
	volumeID string
	attached bool
}

type volumeAttachmentFlags []volumeAttachmentFlag

func (va volumeAttachmentFlags) Len() int {
	return len(va)
}

func (va volumeAttachmentFlags) Swap(i, j int) {
	va[i], va[j] = va[j], va[i]
}

func (va volumeAttachmentFlags) Less(i, j int) bool {
	if va[i].volumeID < va[j].volumeID {
		return true
	}
	if va[i].volumeID > va[j].volumeID {
		return false
	}
	return va[j].attached
}

func serializeAttachments(attachments map[*volume.Spec]bool) string {
	var attachmentFlags volumeAttachmentFlags
	for spec, attached := range attachments {
		attachmentFlags = append(attachmentFlags, volumeAttachmentFlag{spec.Name(), attached})
	}
	sort.Sort(attachmentFlags)
	return fmt.Sprint(attachmentFlags)
}

// newPlugin creates a new gcePersistentDiskPlugin with fake cloud, NewAttacher
// and NewDetacher won't work.
func newPlugin(t *testing.T) *cinderPlugin {
	host := volumetest.NewFakeVolumeHost(t, "/tmp", nil, nil)
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
		Volume: &v1.Volume{
			Name: name,
			VolumeSource: v1.VolumeSource{
				Cinder: &v1.CinderVolumeSource{
					VolumeID: name,
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
					Cinder: &v1.CinderPersistentVolumeSource{
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
	instanceID    string
	volumeID      string
	retDeviceName string
	ret           error
}

type detachCall struct {
	instanceID string
	devicePath string
	ret        error
}

type operationPendingCall struct {
	diskName     string
	pending      bool
	volumeStatus string
	ret          error
}

type diskIsAttachedCall struct {
	instanceID string
	nodeName   types.NodeName
	volumeID   string
	isAttached bool
	ret        error
}

type diskPathCall struct {
	instanceID string
	volumeID   string
	retPath    string
	ret        error
}

type disksAreAttachedCall struct {
	instanceID  string
	nodeName    types.NodeName
	volumeIDs   []string
	areAttached map[string]bool
	ret         error
}

func (testcase *testcase) AttachDisk(instanceID, volumeID string) (string, error) {
	expected := &testcase.attach

	if expected.volumeID == "" && expected.instanceID == "" {
		// testcase.attach looks uninitialized, test did not expect to call
		// AttachDisk
		testcase.t.Errorf("unexpected AttachDisk call")
		return "", errors.New("unexpected AttachDisk call")
	}

	if expected.volumeID != volumeID {
		testcase.t.Errorf("unexpected AttachDisk call: expected volumeID %s, got %s", expected.volumeID, volumeID)
		return "", errors.New("unexpected AttachDisk call: wrong volumeID")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("unexpected AttachDisk call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return "", errors.New("unexpected AttachDisk call: wrong instanceID")
	}

	klog.V(4).Infof("AttachDisk call: %s, %s, returning %q, %v", volumeID, instanceID, expected.retDeviceName, expected.ret)

	testcase.attachOrDetach = &attachStatus
	return expected.retDeviceName, expected.ret
}

func (testcase *testcase) DetachDisk(instanceID, volumeID string) error {
	expected := &testcase.detach

	if expected.devicePath == "" && expected.instanceID == "" {
		// testcase.detach looks uninitialized, test did not expect to call
		// DetachDisk
		testcase.t.Errorf("unexpected DetachDisk call")
		return errors.New("unexpected DetachDisk call")
	}

	if expected.devicePath != volumeID {
		testcase.t.Errorf("unexpected DetachDisk call: expected volumeID %s, got %s", expected.devicePath, volumeID)
		return errors.New("unexpected DetachDisk call: wrong volumeID")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("unexpected DetachDisk call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return errors.New("unexpected DetachDisk call: wrong instanceID")
	}

	klog.V(4).Infof("DetachDisk call: %s, %s, returning %v", volumeID, instanceID, expected.ret)

	testcase.attachOrDetach = &detachStatus
	return expected.ret
}

func (testcase *testcase) OperationPending(diskName string) (bool, string, error) {
	expected := &testcase.operationPending

	if expected.volumeStatus == VolumeStatusPending {
		klog.V(4).Infof("OperationPending call: %s, returning %v, %v, %v", diskName, expected.pending, expected.volumeStatus, expected.ret)
		return true, expected.volumeStatus, expected.ret
	}

	klog.V(4).Infof("OperationPending call: %s, returning %v, %v, %v", diskName, expected.pending, expected.volumeStatus, expected.ret)

	return false, expected.volumeStatus, expected.ret
}

func (testcase *testcase) DiskIsAttached(instanceID, volumeID string) (bool, error) {
	expected := &testcase.diskIsAttached
	// If testcase call DetachDisk*, return false
	if *testcase.attachOrDetach == detachStatus {
		return false, nil
	}

	// If testcase call AttachDisk*, return true
	if *testcase.attachOrDetach == attachStatus {
		return true, nil
	}

	if expected.volumeID == "" && expected.instanceID == "" {
		// testcase.diskIsAttached looks uninitialized, test did not expect to
		// call DiskIsAttached
		testcase.t.Errorf("unexpected DiskIsAttached call")
		return false, errors.New("unexpected DiskIsAttached call")
	}

	if expected.volumeID != volumeID {
		testcase.t.Errorf("unexpected DiskIsAttached call: expected volumeID %s, got %s", expected.volumeID, volumeID)
		return false, errors.New("unexpected DiskIsAttached call: wrong volumeID")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("unexpected DiskIsAttached call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return false, errors.New("unexpected DiskIsAttached call: wrong instanceID")
	}

	klog.V(4).Infof("DiskIsAttached call: %s, %s, returning %v, %v", volumeID, instanceID, expected.isAttached, expected.ret)

	return expected.isAttached, expected.ret
}

func (testcase *testcase) GetAttachmentDiskPath(instanceID, volumeID string) (string, error) {
	expected := &testcase.diskPath
	if expected.volumeID == "" && expected.instanceID == "" {
		// testcase.diskPath looks uninitialized, test did not expect to
		// call GetAttachmentDiskPath
		testcase.t.Errorf("unexpected GetAttachmentDiskPath call")
		return "", errors.New("unexpected GetAttachmentDiskPath call")
	}

	if expected.volumeID != volumeID {
		testcase.t.Errorf("unexpected GetAttachmentDiskPath call: expected volumeID %s, got %s", expected.volumeID, volumeID)
		return "", errors.New("unexpected GetAttachmentDiskPath call: wrong volumeID")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("unexpected GetAttachmentDiskPath call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return "", errors.New("unexpected GetAttachmentDiskPath call: wrong instanceID")
	}

	klog.V(4).Infof("GetAttachmentDiskPath call: %s, %s, returning %v, %v", volumeID, instanceID, expected.retPath, expected.ret)

	return expected.retPath, expected.ret
}

func (testcase *testcase) ShouldTrustDevicePath() bool {
	return true
}

func (testcase *testcase) DiskIsAttachedByName(nodeName types.NodeName, volumeID string) (bool, string, error) {
	expected := &testcase.diskIsAttached
	instanceID := expected.instanceID
	// If testcase call DetachDisk*, return false
	if *testcase.attachOrDetach == detachStatus {
		return false, instanceID, nil
	}

	// If testcase call AttachDisk*, return true
	if *testcase.attachOrDetach == attachStatus {
		return true, instanceID, nil
	}

	if expected.nodeName != nodeName {
		testcase.t.Errorf("unexpected DiskIsAttachedByName call: expected nodename %s, got %s", expected.nodeName, nodeName)
		return false, instanceID, errors.New("unexpected DiskIsAttachedByName call: wrong nodename")
	}

	if expected.volumeID == "" && expected.instanceID == "" {
		// testcase.diskIsAttached looks uninitialized, test did not expect to
		// call DiskIsAttached
		testcase.t.Errorf("unexpected DiskIsAttachedByName call")
		return false, instanceID, errors.New("unexpected DiskIsAttachedByName call")
	}

	if expected.volumeID != volumeID {
		testcase.t.Errorf("unexpected DiskIsAttachedByName call: expected volumeID %s, got %s", expected.volumeID, volumeID)
		return false, instanceID, errors.New("unexpected DiskIsAttachedByName call: wrong volumeID")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("unexpected DiskIsAttachedByName call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return false, instanceID, errors.New("unexpected DiskIsAttachedByName call: wrong instanceID")
	}

	klog.V(4).Infof("DiskIsAttachedByName call: %s, %s, returning %v, %v, %v", volumeID, nodeName, expected.isAttached, expected.instanceID, expected.ret)

	return expected.isAttached, expected.instanceID, expected.ret
}

func (testcase *testcase) CreateVolume(name string, size int, vtype, availability string, tags *map[string]string) (string, string, string, bool, error) {
	return "", "", "", false, errors.New("Not implemented")
}

func (testcase *testcase) GetDevicePath(volumeID string) string {
	return ""
}

func (testcase *testcase) InstanceID() (string, error) {
	return testcase.instanceID, nil
}

func (testcase *testcase) ExpandVolume(volumeID string, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error) {
	return resource.Quantity{}, nil
}

func (testcase *testcase) DeleteVolume(volumeID string) error {
	return errors.New("Not implemented")
}

func (testcase *testcase) GetAutoLabelsForPD(name string) (map[string]string, error) {
	return map[string]string{}, errors.New("Not implemented")
}

func (testcase *testcase) Instances() (cloudprovider.Instances, bool) {
	return &instances{testcase.instanceID}, true
}

func (testcase *testcase) DisksAreAttached(instanceID string, volumeIDs []string) (map[string]bool, error) {
	expected := &testcase.disksAreAttached

	areAttached := make(map[string]bool)

	if len(expected.volumeIDs) == 0 && expected.instanceID == "" {
		// testcase.volumeIDs looks uninitialized, test did not expect to call DisksAreAttached
		testcase.t.Errorf("Unexpected DisksAreAttached call!")
		return areAttached, errors.New("Unexpected DisksAreAttached call")
	}

	if !reflect.DeepEqual(expected.volumeIDs, volumeIDs) {
		testcase.t.Errorf("Unexpected DisksAreAttached call: expected volumeIDs %v, got %v", expected.volumeIDs, volumeIDs)
		return areAttached, errors.New("Unexpected DisksAreAttached call: wrong volumeID")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected DisksAreAttached call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return areAttached, errors.New("Unexpected DisksAreAttached call: wrong instanceID")
	}

	klog.V(4).Infof("DisksAreAttached call: %v, %s, returning %v, %v", volumeIDs, instanceID, expected.areAttached, expected.ret)

	return expected.areAttached, expected.ret
}

func (testcase *testcase) DisksAreAttachedByName(nodeName types.NodeName, volumeIDs []string) (map[string]bool, error) {
	expected := &testcase.disksAreAttached
	areAttached := make(map[string]bool)

	instanceID := expected.instanceID
	if expected.nodeName != nodeName {
		testcase.t.Errorf("Unexpected DisksAreAttachedByName call: expected nodeName %s, got %s", expected.nodeName, nodeName)
		return areAttached, errors.New("Unexpected DisksAreAttachedByName call: wrong nodename")
	}
	if len(expected.volumeIDs) == 0 && expected.instanceID == "" {
		// testcase.volumeIDs looks uninitialized, test did not expect to call DisksAreAttached
		testcase.t.Errorf("Unexpected DisksAreAttachedByName call!")
		return areAttached, errors.New("Unexpected DisksAreAttachedByName call")
	}

	if !reflect.DeepEqual(expected.volumeIDs, volumeIDs) {
		testcase.t.Errorf("Unexpected DisksAreAttachedByName call: expected volumeIDs %v, got %v", expected.volumeIDs, volumeIDs)
		return areAttached, errors.New("Unexpected DisksAreAttachedByName call: wrong volumeID")
	}

	if expected.instanceID != instanceID {
		testcase.t.Errorf("Unexpected DisksAreAttachedByName call: expected instanceID %s, got %s", expected.instanceID, instanceID)
		return areAttached, errors.New("Unexpected DisksAreAttachedByName call: wrong instanceID")
	}

	klog.V(4).Infof("DisksAreAttachedByName call: %v, %s, returning %v, %v", volumeIDs, nodeName, expected.areAttached, expected.ret)

	return expected.areAttached, expected.ret
}

// Implementation of fake cloudprovider.Instances
type instances struct {
	instanceID string
}

func (instances *instances) NodeAddresses(ctx context.Context, name types.NodeName) ([]v1.NodeAddress, error) {
	return []v1.NodeAddress{}, errors.New("Not implemented")
}

func (instances *instances) NodeAddressesByProviderID(ctx context.Context, providerID string) ([]v1.NodeAddress, error) {
	return []v1.NodeAddress{}, errors.New("Not implemented")
}

func (instances *instances) InstanceID(ctx context.Context, name types.NodeName) (string, error) {
	return instances.instanceID, nil
}

func (instances *instances) InstanceType(ctx context.Context, name types.NodeName) (string, error) {
	return "", errors.New("Not implemented")
}

func (instances *instances) InstanceTypeByProviderID(ctx context.Context, providerID string) (string, error) {
	return "", errors.New("Not implemented")
}

func (instances *instances) InstanceExistsByProviderID(ctx context.Context, providerID string) (bool, error) {
	return false, errors.New("unimplemented")
}

func (instances *instances) InstanceShutdownByProviderID(ctx context.Context, providerID string) (bool, error) {
	return false, errors.New("unimplemented")
}

func (instances *instances) List(filter string) ([]types.NodeName, error) {
	return []types.NodeName{}, errors.New("Not implemented")
}

func (instances *instances) AddSSHKeyToAllInstances(ctx context.Context, user string, keyData []byte) error {
	return cloudprovider.NotImplemented
}

func (instances *instances) CurrentNodeName(ctx context.Context, hostname string) (types.NodeName, error) {
	return "", errors.New("Not implemented")
}
