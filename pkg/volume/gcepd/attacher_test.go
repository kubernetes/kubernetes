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

package gcepd

import (
	"errors"
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudprovider "k8s.io/cloud-provider"
	cloudvolume "k8s.io/cloud-provider/volume"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/legacy-cloud-providers/gce"

	"strings"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
)

func TestGetDeviceName_Volume(t *testing.T) {
	plugin := newPlugin(t)
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
	plugin := newPlugin(t)
	name := "my-pd-pv"
	spec := createPVSpec(name, true, nil)

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

func TestAttachDetachRegional(t *testing.T) {
	diskName := "disk"
	nodeName := types.NodeName("instance")
	readOnly := false
	regional := true
	spec := createPVSpec(diskName, readOnly, []string{"zone1", "zone2"})
	// Successful Attach call
	testcase := testcase{
		name:           "Attach_Regional_Positive",
		diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName: {diskName}}, nil},
		attach:         attachCall{diskName, nodeName, readOnly, regional, nil},
		test: func(testcase *testcase) error {
			attacher := newAttacher(testcase)
			devicePath, err := attacher.Attach(spec, nodeName)
			if devicePath != "/dev/disk/by-id/google-disk" {
				return fmt.Errorf("devicePath incorrect. Expected<\"/dev/disk/by-id/google-disk\"> Actual: <%q>", devicePath)
			}
			return err
		},
	}

	err := testcase.test(&testcase)
	if err != testcase.expectedReturn {
		t.Errorf("%s failed: expected err=%v, got %v", testcase.name, testcase.expectedReturn, err)
	}
}

func TestAttachDetach(t *testing.T) {
	diskName := "disk"
	nodeName := types.NodeName("instance")
	readOnly := false
	regional := false
	spec := createVolSpec(diskName, readOnly)
	attachError := errors.New("Fake attach error")
	detachError := errors.New("Fake detach error")
	diskCheckError := errors.New("Fake DiskIsAttached error")
	tests := []testcase{
		// Successful Attach call
		{
			name:           "Attach_Positive",
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName: {}}, nil},
			attach:         attachCall{diskName, nodeName, readOnly, regional, nil},
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
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName: {diskName}}, nil},
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
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName: {}}, diskCheckError},
			attach:         attachCall{diskName, nodeName, readOnly, regional, nil},
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
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName: {}}, diskCheckError},
			attach:         attachCall{diskName, nodeName, readOnly, regional, attachError},
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
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName: {diskName}}, nil},
			detach:         detachCall{diskName, nodeName, nil},
			test: func(testcase *testcase) error {
				detacher := newDetacher(testcase)
				return detacher.Detach(diskName, nodeName)
			},
		},

		// Disk is already detached
		{
			name:           "Detach_Positive_AlreadyDetached",
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName: {}}, nil},
			test: func(testcase *testcase) error {
				detacher := newDetacher(testcase)
				return detacher.Detach(diskName, nodeName)
			},
		},

		// Detach succeeds when DiskIsAttached fails
		{
			name:           "Detach_Positive_CheckFails",
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName: {}}, diskCheckError},
			detach:         detachCall{diskName, nodeName, nil},
			test: func(testcase *testcase) error {
				detacher := newDetacher(testcase)
				return detacher.Detach(diskName, nodeName)
			},
		},

		// Detach fails
		{
			name:           "Detach_Negative",
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName: {}}, diskCheckError},
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
			t.Errorf("%s failed: expected err=%v, got %v", testcase.name, testcase.expectedReturn, err)
		}
	}
}

func TestVerifyVolumesAttached(t *testing.T) {
	readOnly := false
	nodeName1 := types.NodeName("instance1")
	nodeName2 := types.NodeName("instance2")

	diskAName := "diskA"
	diskBName := "diskB"
	diskCName := "diskC"
	diskASpec := createVolSpec(diskAName, readOnly)
	diskBSpec := createVolSpec(diskBName, readOnly)
	diskCSpec := createVolSpec(diskCName, readOnly)

	verifyDiskAttachedInResult := func(results map[*volume.Spec]bool, spec *volume.Spec, expected bool) error {
		found, ok := results[spec]
		if !ok {
			return fmt.Errorf("expected to find volume %s in verifcation result, but didn't", spec.Name())
		}
		if found != expected {
			return fmt.Errorf("expected to find volume %s to be have attached value %v but got %v", spec.Name(), expected, found)
		}
		return nil
	}

	tests := []testcase{
		// Successful VolumesAreAttached
		{
			name:           "VolumesAreAttached_Positive",
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName1: {diskAName, diskBName}}, nil},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				results, err := attacher.VolumesAreAttached([]*volume.Spec{diskASpec, diskBSpec}, nodeName1)
				if err != nil {
					return err
				}
				err = verifyDiskAttachedInResult(results, diskASpec, true)
				if err != nil {
					return err
				}
				return verifyDiskAttachedInResult(results, diskBSpec, true)
			},
		},

		// Successful VolumesAreAttached for detached disk
		{
			name:           "VolumesAreAttached_Negative",
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName1: {diskAName}}, nil},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				results, err := attacher.VolumesAreAttached([]*volume.Spec{diskASpec, diskBSpec}, nodeName1)
				if err != nil {
					return err
				}
				err = verifyDiskAttachedInResult(results, diskASpec, true)
				if err != nil {
					return err
				}
				return verifyDiskAttachedInResult(results, diskBSpec, false)
			},
		},

		// VolumesAreAttached with InstanceNotFound
		{
			name:           "VolumesAreAttached_InstanceNotFound",
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{}, nil},
			expectedReturn: cloudprovider.InstanceNotFound,
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				_, err := attacher.VolumesAreAttached([]*volume.Spec{diskASpec}, nodeName1)
				if err != cloudprovider.InstanceNotFound {
					return fmt.Errorf("expected InstanceNotFound error, but got %v", err)
				}
				return err
			},
		},

		// Successful BulkDisksAreAttached
		{
			name:           "BulkDisksAreAttached_Positive",
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName1: {diskAName}, nodeName2: {diskBName, diskCName}}, nil},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				results, err := attacher.BulkVerifyVolumes(map[types.NodeName][]*volume.Spec{nodeName1: {diskASpec}, nodeName2: {diskBSpec, diskCSpec}})
				if err != nil {
					return err
				}
				disksAttachedNode1, nodeFound := results[nodeName1]
				if !nodeFound {
					return fmt.Errorf("expected to find node %s but didn't", nodeName1)
				}
				if err := verifyDiskAttachedInResult(disksAttachedNode1, diskASpec, true); err != nil {
					return err
				}
				disksAttachedNode2, nodeFound := results[nodeName2]
				if !nodeFound {
					return fmt.Errorf("expected to find node %s but didn't", nodeName2)
				}
				if err := verifyDiskAttachedInResult(disksAttachedNode2, diskBSpec, true); err != nil {
					return err
				}
				return verifyDiskAttachedInResult(disksAttachedNode2, diskCSpec, true)
			},
		},

		// Successful BulkDisksAreAttached for detached disk
		{
			name:           "BulkDisksAreAttached_Negative",
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName1: {}, nodeName2: {diskBName}}, nil},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				results, err := attacher.BulkVerifyVolumes(map[types.NodeName][]*volume.Spec{nodeName1: {diskASpec}, nodeName2: {diskBSpec, diskCSpec}})
				if err != nil {
					return err
				}
				disksAttachedNode1, nodeFound := results[nodeName1]
				if !nodeFound {
					return fmt.Errorf("expected to find node %s but didn't", nodeName1)
				}
				if err := verifyDiskAttachedInResult(disksAttachedNode1, diskASpec, false); err != nil {
					return err
				}
				disksAttachedNode2, nodeFound := results[nodeName2]
				if !nodeFound {
					return fmt.Errorf("expected to find node %s but didn't", nodeName2)
				}
				if err := verifyDiskAttachedInResult(disksAttachedNode2, diskBSpec, true); err != nil {
					return err
				}
				return verifyDiskAttachedInResult(disksAttachedNode2, diskCSpec, false)
			},
		},

		// Successful BulkDisksAreAttached with InstanceNotFound
		{
			name:           "BulkDisksAreAttached_InstanceNotFound",
			diskIsAttached: diskIsAttachedCall{disksAttachedMap{nodeName1: {diskAName}}, nil},
			test: func(testcase *testcase) error {
				attacher := newAttacher(testcase)
				results, err := attacher.BulkVerifyVolumes(map[types.NodeName][]*volume.Spec{nodeName1: {diskASpec}, nodeName2: {diskBSpec, diskCSpec}})
				if err != nil {
					return err
				}
				disksAttachedNode1, nodeFound := results[nodeName1]
				if !nodeFound {
					return fmt.Errorf("expected to find node %s but didn't", nodeName1)
				}
				if err := verifyDiskAttachedInResult(disksAttachedNode1, diskASpec, true); err != nil {
					return err
				}
				disksAttachedNode2, nodeFound := results[nodeName2]
				if !nodeFound {
					return fmt.Errorf("expected to find node %s but didn't", nodeName2)
				}
				if err := verifyDiskAttachedInResult(disksAttachedNode2, diskBSpec, false); err != nil {
					return err
				}
				return verifyDiskAttachedInResult(disksAttachedNode2, diskCSpec, false)
			},
		},
	}

	for _, testcase := range tests {
		testcase.t = t
		err := testcase.test(&testcase)
		if err != testcase.expectedReturn {
			t.Errorf("%s failed: expected err=%v, got %v", testcase.name, testcase.expectedReturn, err)
		}
	}
}

// newPlugin creates a new gcePersistentDiskPlugin with fake cloud, NewAttacher
// and NewDetacher won't work.
func newPlugin(t *testing.T) *gcePersistentDiskPlugin {
	host := volumetest.NewFakeVolumeHost(t,
		"/tmp", /* rootDir */
		nil,    /* kubeClient */
		nil,    /* plugins */
	)
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

func createPVSpec(name string, readOnly bool, zones []string) *volume.Spec {
	spec := &volume.Spec{
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

	if zones != nil {
		zonesLabel := strings.Join(zones, cloudvolume.LabelMultiZoneDelimiter)
		spec.PersistentVolume.ObjectMeta.Labels = map[string]string{
			v1.LabelZoneFailureDomain: zonesLabel,
		}
	}

	return spec
}

// Fake GCE implementation

type attachCall struct {
	diskName string
	nodeName types.NodeName
	readOnly bool
	regional bool
	retErr   error
}

type detachCall struct {
	devicePath string
	nodeName   types.NodeName
	retErr     error
}

type diskIsAttachedCall struct {
	attachedDisks disksAttachedMap
	retErr        error
}

// disksAttachedMap specifies what disks in the test scenario are actually attached to each node
type disksAttachedMap map[types.NodeName][]string

func (testcase *testcase) AttachDisk(diskName string, nodeName types.NodeName, readOnly bool, regional bool) error {
	expected := &testcase.attach

	if expected.diskName == "" && expected.nodeName == "" {
		// testcase.attach looks uninitialized, test did not expect to call AttachDisk
		return errors.New("unexpected AttachDisk call")
	}

	if expected.diskName != diskName {
		return fmt.Errorf("Unexpected AttachDisk call: expected diskName %s, got %s", expected.diskName, diskName)
	}

	if expected.nodeName != nodeName {
		return fmt.Errorf("Unexpected AttachDisk call: expected nodeName %s, got %s", expected.nodeName, nodeName)
	}

	if expected.readOnly != readOnly {
		return fmt.Errorf("Unexpected AttachDisk call: expected readOnly %v, got %v", expected.readOnly, readOnly)
	}

	if expected.regional != regional {
		return fmt.Errorf("Unexpected AttachDisk call: expected regional %v, got %v", expected.regional, regional)
	}

	klog.V(4).Infof("AttachDisk call: %s, %s, %v, returning %v", diskName, nodeName, readOnly, expected.retErr)

	return expected.retErr
}

func (testcase *testcase) DetachDisk(devicePath string, nodeName types.NodeName) error {
	expected := &testcase.detach

	if expected.devicePath == "" && expected.nodeName == "" {
		// testcase.detach looks uninitialized, test did not expect to call DetachDisk
		return errors.New("unexpected DetachDisk call")
	}

	if expected.devicePath != devicePath {
		return fmt.Errorf("Unexpected DetachDisk call: expected devicePath %s, got %s", expected.devicePath, devicePath)
	}

	if expected.nodeName != nodeName {
		return fmt.Errorf("Unexpected DetachDisk call: expected nodeName %s, got %s", expected.nodeName, nodeName)
	}

	klog.V(4).Infof("DetachDisk call: %s, %s, returning %v", devicePath, nodeName, expected.retErr)

	return expected.retErr
}

func (testcase *testcase) DiskIsAttached(diskName string, nodeName types.NodeName) (bool, error) {
	expected := &testcase.diskIsAttached

	if expected.attachedDisks == nil {
		// testcase.attachedDisks looks uninitialized, test did not expect to call DiskIsAttached
		return false, errors.New("unexpected DiskIsAttached call")
	}

	if expected.retErr != nil {
		return false, expected.retErr
	}

	disksForNode, nodeExists := expected.attachedDisks[nodeName]
	if !nodeExists {
		return false, cloudprovider.InstanceNotFound
	}

	found := false
	for _, diskAttachedName := range disksForNode {
		if diskAttachedName == diskName {
			found = true
		}
	}
	klog.V(4).Infof("DiskIsAttached call: %s, %s, returning %v", diskName, nodeName, found)
	return found, nil
}

func (testcase *testcase) DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error) {
	verifiedDisks := make(map[string]bool)
	for _, name := range diskNames {
		found, err := testcase.DiskIsAttached(name, nodeName)
		if err != nil {
			return nil, err
		}
		verifiedDisks[name] = found
	}
	return verifiedDisks, nil
}

func (testcase *testcase) BulkDisksAreAttached(diskByNodes map[types.NodeName][]string) (map[types.NodeName]map[string]bool, error) {
	verifiedDisksByNodes := make(map[types.NodeName]map[string]bool)
	for nodeName, disksForNode := range diskByNodes {
		verifiedDisks, err := testcase.DisksAreAttached(disksForNode, nodeName)
		if err != nil {
			if err != cloudprovider.InstanceNotFound {
				return nil, err
			}
			verifiedDisks = make(map[string]bool)
			for _, diskName := range disksForNode {
				verifiedDisks[diskName] = false
			}
		}
		verifiedDisksByNodes[nodeName] = verifiedDisks
	}

	return verifiedDisksByNodes, nil
}

func (testcase *testcase) CreateDisk(name string, diskType string, zone string, sizeGb int64, tags map[string]string) (*gce.Disk, error) {
	return nil, errors.New("Not implemented")
}

func (testcase *testcase) CreateRegionalDisk(name string, diskType string, replicaZones sets.String, sizeGb int64, tags map[string]string) (*gce.Disk, error) {
	return nil, errors.New("Not implemented")
}

func (testcase *testcase) DeleteDisk(diskToDelete string) error {
	return errors.New("Not implemented")
}

func (testcase *testcase) GetAutoLabelsForPD(*gce.Disk) (map[string]string, error) {
	return map[string]string{}, errors.New("Not implemented")
}

func (testcase *testcase) ResizeDisk(
	diskName string,
	oldSize resource.Quantity,
	newSize resource.Quantity) (resource.Quantity, error) {
	return oldSize, errors.New("Not implemented")
}
