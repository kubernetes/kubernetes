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

package cache

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
)

// Calls AddVolumeNode() once.
// Verifies a single volume/node entry exists.
func Test_AddVolumeNode_Positive_NewVolumeNewNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)

	nodeName := "node-name"
	devicePath := "fake/device/path"

	// Act
	generatedVolumeName, err := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)

	// Assert
	if err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", err)
	}

	volumeNodeComboExists := asw.VolumeNodeExists(generatedVolumeName, nodeName)
	if !volumeNodeComboExists {
		t.Fatalf("%q/%q volume/node combo does not exist, it should.", generatedVolumeName, nodeName)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Calls AddVolumeNode() twice. Second time use a different node name.
// Verifies two volume/node entries exist with the same volumeSpec.
func Test_AddVolumeNode_Positive_ExistingVolumeNewNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	node1Name := "node1-name"
	node2Name := "node2-name"
	devicePath := "fake/device/path"

	// Act
	generatedVolumeName1, add1Err := asw.AddVolumeNode(volumeSpec, node1Name, devicePath)
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volumeSpec, node2Name, devicePath)

	// Assert
	if add1Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add1Err)
	}
	if add2Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add2Err)
	}

	if generatedVolumeName1 != generatedVolumeName2 {
		t.Fatalf(
			"Generated volume names for the same volume should be the same but they are not: %q and %q",
			generatedVolumeName1,
			generatedVolumeName2)
	}

	volumeNode1ComboExists := asw.VolumeNodeExists(generatedVolumeName1, node1Name)
	if !volumeNode1ComboExists {
		t.Fatalf("%q/%q volume/node combo does not exist, it should.", generatedVolumeName1, node1Name)
	}

	volumeNode2ComboExists := asw.VolumeNodeExists(generatedVolumeName1, node2Name)
	if !volumeNode2ComboExists {
		t.Fatalf("%q/%q volume/node combo does not exist, it should.", generatedVolumeName1, node2Name)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 2 {
		t.Fatalf("len(attachedVolumes) Expected: <2> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, string(volumeName), node1Name, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, string(volumeName), node2Name, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Calls AddVolumeNode() twice. Uses the same volume and node both times.
// Verifies a single volume/node entry exists.
func Test_AddVolumeNode_Positive_ExistingVolumeExistingNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"

	// Act
	generatedVolumeName1, add1Err := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)

	// Assert
	if add1Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add1Err)
	}
	if add2Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add2Err)
	}

	if generatedVolumeName1 != generatedVolumeName2 {
		t.Fatalf(
			"Generated volume names for the same volume should be the same but they are not: %q and %q",
			generatedVolumeName1,
			generatedVolumeName2)
	}

	volumeNodeComboExists := asw.VolumeNodeExists(generatedVolumeName1, nodeName)
	if !volumeNodeComboExists {
		t.Fatalf("%q/%q volume/node combo does not exist, it should.", generatedVolumeName1, nodeName)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, string(volumeName), nodeName, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume/node entry.
// Calls DeleteVolumeNode() to delete volume/node.
// Verifies no volume/node entries exists.
func Test_DeleteVolumeNode_Positive_VolumeExistsNodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	asw.DeleteVolumeNode(generatedVolumeName, nodeName)

	// Assert
	volumeNodeComboExists := asw.VolumeNodeExists(generatedVolumeName, nodeName)
	if volumeNodeComboExists {
		t.Fatalf("%q/%q volume/node combo exists, it should not.", generatedVolumeName, nodeName)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 0 {
		t.Fatalf("len(attachedVolumes) Expected: <0> Actual: <%v>", len(attachedVolumes))
	}
}

// Calls DeleteVolumeNode() to delete volume/node on empty data struct
// Verifies no volume/node entries exists.
func Test_DeleteVolumeNode_Positive_VolumeDoesntExistNodeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	nodeName := "node-name"

	// Act
	asw.DeleteVolumeNode(volumeName, nodeName)

	// Assert
	volumeNodeComboExists := asw.VolumeNodeExists(volumeName, nodeName)
	if volumeNodeComboExists {
		t.Fatalf("%q/%q volume/node combo exists, it should not.", volumeName, nodeName)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 0 {
		t.Fatalf("len(attachedVolumes) Expected: <0> Actual: <%v>", len(attachedVolumes))
	}
}

// Populates data struct with two volume/node entries the second one using a
// different node.
// Calls DeleteVolumeNode() to delete first volume/node.
// Verifies only second volume/node entry exists.
func Test_DeleteVolumeNode_Positive_TwoNodesOneDeleted(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	node1Name := "node1-name"
	node2Name := "node2-name"
	devicePath := "fake/device/path"
	generatedVolumeName1, add1Err := asw.AddVolumeNode(volumeSpec, node1Name, devicePath)
	if add1Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add1Err)
	}
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volumeSpec, node2Name, devicePath)
	if add2Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add2Err)
	}
	if generatedVolumeName1 != generatedVolumeName2 {
		t.Fatalf(
			"Generated volume names for the same volume should be the same but they are not: %q and %q",
			generatedVolumeName1,
			generatedVolumeName2)
	}

	// Act
	asw.DeleteVolumeNode(generatedVolumeName1, node1Name)

	// Assert
	volumeNodeComboExists := asw.VolumeNodeExists(generatedVolumeName1, node1Name)
	if volumeNodeComboExists {
		t.Fatalf("%q/%q volume/node combo exists, it should not.", generatedVolumeName1, node1Name)
	}

	volumeNodeComboExists = asw.VolumeNodeExists(generatedVolumeName1, node2Name)
	if !volumeNodeComboExists {
		t.Fatalf("%q/%q volume/node combo does not exist, it should.", generatedVolumeName1, node2Name)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, string(volumeName), node2Name, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume/node entry.
// Calls VolumeNodeExists() to verify entry.
// Verifies the populated volume/node entry exists.
func Test_VolumeNodeExists_Positive_VolumeExistsNodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	volumeNodeComboExists := asw.VolumeNodeExists(generatedVolumeName, nodeName)

	// Assert
	if !volumeNodeComboExists {
		t.Fatalf("%q/%q volume/node combo does not exist, it should.", generatedVolumeName, nodeName)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume1/node1 entry.
// Calls VolumeNodeExists() with volume1/node2.
// Verifies requested entry does not exist, but populated entry does.
func Test_VolumeNodeExists_Positive_VolumeExistsNodeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	node1Name := "node1-name"
	node2Name := "node2-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, node1Name, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	volumeNodeComboExists := asw.VolumeNodeExists(generatedVolumeName, node2Name)

	// Assert
	if volumeNodeComboExists {
		t.Fatalf("%q/%q volume/node combo exists, it should not.", generatedVolumeName, node2Name)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), node1Name, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Calls VolumeNodeExists() on empty data struct.
// Verifies requested entry does not exist.
func Test_VolumeNodeExists_Positive_VolumeAndNodeDontExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	nodeName := "node-name"

	// Act
	volumeNodeComboExists := asw.VolumeNodeExists(volumeName, nodeName)

	// Assert
	if volumeNodeComboExists {
		t.Fatalf("%q/%q volume/node combo exists, it should not.", volumeName, nodeName)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 0 {
		t.Fatalf("len(attachedVolumes) Expected: <0> Actual: <%v>", len(attachedVolumes))
	}
}

// Calls GetAttachedVolumes() on empty data struct.
// Verifies no volume/node entries are returned.
func Test_GetAttachedVolumes_Positive_NoVolumesOrNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)

	// Act
	attachedVolumes := asw.GetAttachedVolumes()

	// Assert
	if len(attachedVolumes) != 0 {
		t.Fatalf("len(attachedVolumes) Expected: <0> Actual: <%v>", len(attachedVolumes))
	}
}

// Populates data struct with one volume/node entry.
// Calls GetAttachedVolumes() to get list of entries.
// Verifies one volume/node entry is returned.
func Test_GetAttachedVolumes_Positive_OneVolumeOneNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	attachedVolumes := asw.GetAttachedVolumes()

	// Assert
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with two volume/node entries (different node and volume).
// Calls GetAttachedVolumes() to get list of entries.
// Verifies both volume/node entries are returned.
func Test_GetAttachedVolumes_Positive_TwoVolumeTwoNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volume1Name := api.UniqueVolumeName("volume1-name")
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(string(volume1Name), volume1Name)
	node1Name := "node1-name"
	devicePath := "fake/device/path"
	generatedVolumeName1, add1Err := asw.AddVolumeNode(volume1Spec, node1Name, devicePath)
	if add1Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add1Err)
	}
	volume2Name := api.UniqueVolumeName("volume2-name")
	volume2Spec := controllervolumetesting.GetTestVolumeSpec(string(volume2Name), volume2Name)
	node2Name := "node2-name"
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volume2Spec, node2Name, devicePath)
	if add2Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add2Err)
	}

	// Act
	attachedVolumes := asw.GetAttachedVolumes()

	// Assert
	if len(attachedVolumes) != 2 {
		t.Fatalf("len(attachedVolumes) Expected: <2> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, string(volume1Name), node1Name, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName2, string(volume2Name), node2Name, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with two volume/node entries (same volume different node).
// Calls GetAttachedVolumes() to get list of entries.
// Verifies both volume/node entries are returned.
func Test_GetAttachedVolumes_Positive_OneVolumeTwoNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	node1Name := "node1-name"
	devicePath := "fake/device/path"
	generatedVolumeName1, add1Err := asw.AddVolumeNode(volumeSpec, node1Name, devicePath)
	if add1Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add1Err)
	}
	node2Name := "node2-name"
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volumeSpec, node2Name, devicePath)
	if add2Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add2Err)
	}

	if generatedVolumeName1 != generatedVolumeName2 {
		t.Fatalf(
			"Generated volume names for the same volume should be the same but they are not: %q and %q",
			generatedVolumeName1,
			generatedVolumeName2)
	}

	// Act
	attachedVolumes := asw.GetAttachedVolumes()

	// Assert
	if len(attachedVolumes) != 2 {
		t.Fatalf("len(attachedVolumes) Expected: <2> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, string(volumeName), node1Name, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, string(volumeName), node2Name, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume/node entry.
// Verifies mountedByNode is true and DetachRequestedTime is zero.
func Test_SetVolumeMountedByNode_Positive_Set(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act: do not mark -- test default value

	// Assert
	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume/node entry.
// Calls SetVolumeMountedByNode twice, first setting mounted to true then false.
// Verifies mountedByNode is false.
func Test_SetVolumeMountedByNode_Positive_UnsetWithInitialSet(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	setVolumeMountedErr1 := asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, true /* mounted */)
	setVolumeMountedErr2 := asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, false /* mounted */)

	// Assert
	if setVolumeMountedErr1 != nil {
		t.Fatalf("SetVolumeMountedByNode1 failed. Expected <no error> Actual: <%v>", setVolumeMountedErr1)
	}
	if setVolumeMountedErr2 != nil {
		t.Fatalf("SetVolumeMountedByNode2 failed. Expected <no error> Actual: <%v>", setVolumeMountedErr2)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, false /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume/node entry.
// Calls SetVolumeMountedByNode once, setting mounted to false.
// Verifies mountedByNode is still true (since there was no SetVolumeMountedByNode to true call first)
func Test_SetVolumeMountedByNode_Positive_UnsetWithoutInitialSet(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	setVolumeMountedErr := asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, false /* mounted */)

	// Assert
	if setVolumeMountedErr != nil {
		t.Fatalf("SetVolumeMountedByNode failed. Expected <no error> Actual: <%v>", setVolumeMountedErr)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume/node entry.
// Calls SetVolumeMountedByNode twice, first setting mounted to true then false.
// Calls AddVolumeNode to readd the same volume/node.
// Verifies mountedByNode is false and detachRequestedTime is zero.
func Test_SetVolumeMountedByNode_Positive_UnsetWithInitialSetAddVolumeNodeNotReset(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	setVolumeMountedErr1 := asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, true /* mounted */)
	setVolumeMountedErr2 := asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, false /* mounted */)
	generatedVolumeName, addErr = asw.AddVolumeNode(volumeSpec, nodeName, devicePath)

	// Assert
	if setVolumeMountedErr1 != nil {
		t.Fatalf("SetVolumeMountedByNode1 failed. Expected <no error> Actual: <%v>", setVolumeMountedErr1)
	}
	if setVolumeMountedErr2 != nil {
		t.Fatalf("SetVolumeMountedByNode2 failed. Expected <no error> Actual: <%v>", setVolumeMountedErr2)
	}
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, false /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume/node entry.
// Calls RemoveVolumeFromReportAsAttached() once on volume/node entry.
// Calls SetVolumeMountedByNode() twice, first setting mounted to true then false.
// Verifies mountedByNode is false and detachRequestedTime is NOT zero.
func Test_SetVolumeMountedByNode_Positive_UnsetWithInitialSetVerifyDetachRequestedTimePerserved(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}
	_, err := asw.SetDetachRequestTime(generatedVolumeName, nodeName)
	if err != nil {
		t.Fatalf("SetDetachRequestTime failed. Expected: <no error> Actual: <%v>", err)
	}
	err = asw.RemoveVolumeFromReportAsAttached(generatedVolumeName, nodeName)
	if err != nil {
		t.Fatalf("RemoveVolumeFromReportAsAttached failed. Expected: <no error> Actual: <%v>", err)
	}
	expectedDetachRequestedTime := asw.GetAttachedVolumes()[0].DetachRequestedTime

	// Act
	setVolumeMountedErr1 := asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, true /* mounted */)
	setVolumeMountedErr2 := asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, false /* mounted */)

	// Assert
	if setVolumeMountedErr1 != nil {
		t.Fatalf("SetVolumeMountedByNode1 failed. Expected <no error> Actual: <%v>", setVolumeMountedErr1)
	}
	if setVolumeMountedErr2 != nil {
		t.Fatalf("SetVolumeMountedByNode2 failed. Expected <no error> Actual: <%v>", setVolumeMountedErr2)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, false /* expectedMountedByNode */, true /* expectNonZeroDetachRequestedTime */)
	if !expectedDetachRequestedTime.Equal(attachedVolumes[0].DetachRequestedTime) {
		t.Fatalf("DetachRequestedTime changed. Expected: <%v> Actual: <%v>", expectedDetachRequestedTime, attachedVolumes[0].DetachRequestedTime)
	}
}

// Populates data struct with one volume/node entry.
// Verifies mountedByNode is true and detachRequestedTime is zero (default values).
func Test_RemoveVolumeFromReportAsAttached_Positive_Set(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	devicePath := "fake/device/path"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act: do not mark -- test default value

	// Assert
	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume/node entry.
// Calls RemoveVolumeFromReportAsAttached() once on volume/node entry.
// Verifies mountedByNode is true and detachRequestedTime is NOT zero.
func Test_RemoveVolumeFromReportAsAttached_Positive_Marked(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	_, err := asw.SetDetachRequestTime(generatedVolumeName, nodeName)
	if err != nil {
		t.Fatalf("SetDetachRequestTime failed. Expected: <no error> Actual: <%v>", err)
	}
	markDesireToDetachErr := asw.RemoveVolumeFromReportAsAttached(generatedVolumeName, nodeName)
	if markDesireToDetachErr != nil {
		t.Fatalf("MarkDesireToDetach failed. Expected: <no error> Actual: <%v>", markDesireToDetachErr)
	}

	// Assert
	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, true /* expectedMountedByNode */, true /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume/node entry.
// Calls MarkDesireToDetach() once on volume/node entry.
// Calls ResetDetachRequestTime() to reset the detach request time value back to 0.
// Verifies mountedByNode is true and detachRequestedTime is reset to zero.
func Test_MarkDesireToDetach_Positive_MarkedAddVolumeNodeReset(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	_, err := asw.SetDetachRequestTime(generatedVolumeName, nodeName)
	if err != nil {
		t.Fatalf("SetDetachRequestTime failed. Expected: <no error> Actual: <%v>", err)
	}
	markDesireToDetachErr := asw.RemoveVolumeFromReportAsAttached(generatedVolumeName, nodeName)
	// Reset detach request time to 0
	asw.ResetDetachRequestTime(generatedVolumeName, nodeName)

	// Assert
	if markDesireToDetachErr != nil {
		t.Fatalf("RemoveVolumeFromReportAsAttached failed. Expected: <no error> Actual: <%v>", markDesireToDetachErr)
	}
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Assert
	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume/node entry.
// Calls SetVolumeMountedByNode() twice, first setting mounted to true then false.
// Calls RemoveVolumeFromReportAsAttached() once on volume/node entry.
// Verifies mountedByNode is false and detachRequestedTime is NOT zero.
func Test_RemoveVolumeFromReportAsAttached_Positive_UnsetWithInitialSetVolumeMountedByNodePreserved(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}
	setVolumeMountedErr1 := asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, true /* mounted */)
	setVolumeMountedErr2 := asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, false /* mounted */)
	if setVolumeMountedErr1 != nil {
		t.Fatalf("SetVolumeMountedByNode1 failed. Expected <no error> Actual: <%v>", setVolumeMountedErr1)
	}
	if setVolumeMountedErr2 != nil {
		t.Fatalf("SetVolumeMountedByNode2 failed. Expected <no error> Actual: <%v>", setVolumeMountedErr2)
	}

	// Act
	_, err := asw.SetDetachRequestTime(generatedVolumeName, nodeName)
	if err != nil {
		t.Fatalf("SetDetachRequestTime failed. Expected: <no error> Actual: <%v>", err)
	}
	removeVolumeDetachErr := asw.RemoveVolumeFromReportAsAttached(generatedVolumeName, nodeName)
	if removeVolumeDetachErr != nil {
		t.Fatalf("RemoveVolumeFromReportAsAttached failed. Expected: <no error> Actual: <%v>", removeVolumeDetachErr)
	}

	// Assert
	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, false /* expectedMountedByNode */, true /* expectNonZeroDetachRequestedTime */)
}

// Populates data struct with one volume/node entry.
// Calls RemoveVolumeFromReportAsAttached
// Verifyies there is no valume as reported as attached
func Test_RemoveVolumeFromReportAsAttached(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	removeVolumeDetachErr := asw.RemoveVolumeFromReportAsAttached(generatedVolumeName, nodeName)
	if removeVolumeDetachErr != nil {
		t.Fatalf("RemoveVolumeFromReportAsAttached failed. Expected: <no error> Actual: <%v>", removeVolumeDetachErr)
	}

	reportAsAttachedVolumesMap := asw.GetVolumesToReportAttached()
	volumes, exists := reportAsAttachedVolumesMap[nodeName]
	if !exists {
		t.Fatalf("MarkDesireToDetach_UnmarkDesireToDetach failed. Expected: <node %q exist> Actual: <node does not exist in the reportedAsAttached map", nodeName)
	}
	if len(volumes) > 0 {
		t.Fatalf("len(reportAsAttachedVolumes) Expected: <0> Actual: <%v>", len(volumes))
	}

}

// Populates data struct with one volume/node entry.
// Calls RemoveVolumeFromReportAsAttached
// Calls AddVolumeToReportAsAttached to add volume back as attached
// Verifyies there is one volume as reported as attached
func Test_RemoveVolumeFromReportAsAttached_AddVolumeToReportAsAttached_Positive(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	removeVolumeDetachErr := asw.RemoveVolumeFromReportAsAttached(generatedVolumeName, nodeName)
	if removeVolumeDetachErr != nil {
		t.Fatalf("RemoveVolumeFromReportAsAttached failed. Expected: <no error> Actual: <%v>", removeVolumeDetachErr)
	}

	reportAsAttachedVolumesMap := asw.GetVolumesToReportAttached()
	volumes, exists := reportAsAttachedVolumesMap[nodeName]
	if !exists {
		t.Fatalf("MarkDesireToDetach_UnmarkDesireToDetach failed. Expected: <node %q exist> Actual: <node does not exist in the reportedAsAttached map", nodeName)
	}
	if len(volumes) > 0 {
		t.Fatalf("len(reportAsAttachedVolumes) Expected: <0> Actual: <%v>", len(volumes))
	}

	asw.AddVolumeToReportAsAttached(generatedVolumeName, nodeName)
	reportAsAttachedVolumesMap = asw.GetVolumesToReportAttached()
	volumes, exists = reportAsAttachedVolumesMap[nodeName]
	if !exists {
		t.Fatalf("MarkDesireToDetach_UnmarkDesireToDetach failed. Expected: <node %q exist> Actual: <node does not exist in the reportedAsAttached map", nodeName)
	}
	if len(volumes) != 1 {
		t.Fatalf("len(reportAsAttachedVolumes) Expected: <1> Actual: <%v>", len(volumes))
	}
}

// Populates data struct with one volume/node entry.
// Calls RemoveVolumeFromReportAsAttached
// Calls DeleteVolumeNode
// Calls AddVolumeToReportAsAttached
// Verifyies there is no volume as reported as attached
func Test_RemoveVolumeFromReportAsAttached_AddVolumeToReportAsAttached_Negative(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	removeVolumeDetachErr := asw.RemoveVolumeFromReportAsAttached(generatedVolumeName, nodeName)
	if removeVolumeDetachErr != nil {
		t.Fatalf("RemoveVolumeFromReportAsAttached failed. Expected: <no error> Actual: <%v>", removeVolumeDetachErr)
	}

	reportAsAttachedVolumesMap := asw.GetVolumesToReportAttached()
	volumes, exists := reportAsAttachedVolumesMap[nodeName]
	if !exists {
		t.Fatalf("MarkDesireToDetach_UnmarkDesireToDetach failed. Expected: <node %q exist> Actual: <node does not exist in the reportedAsAttached map", nodeName)
	}
	if len(volumes) > 0 {
		t.Fatalf("len(reportAsAttachedVolumes) Expected: <0> Actual: <%v>", len(volumes))
	}

	asw.DeleteVolumeNode(generatedVolumeName, nodeName)
	asw.AddVolumeToReportAsAttached(generatedVolumeName, nodeName)

	reportAsAttachedVolumesMap = asw.GetVolumesToReportAttached()
	volumes, exists = reportAsAttachedVolumesMap[nodeName]
	if !exists {
		t.Fatalf("MarkDesireToDetach_UnmarkDesireToDetach failed. Expected: <node %q exist> Actual: <node does not exist in the reportedAsAttached map", nodeName)
	}
	if len(volumes) > 0 {
		t.Fatalf("len(reportAsAttachedVolumes) Expected: <0> Actual: <%v>", len(volumes))
	}
}

// Populates data struct with one volume/node entry.
// Calls SetDetachRequestTime twice and sleep maxWaitTime (1 second) in between
// The elapsed time returned from the first SetDetachRequestTime call should be smaller than maxWaitTime
// The elapsed time returned from the second SetDetachRequestTime call should be larger than maxWaitTime
func Test_SetDetachRequestTime_Positive(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	maxWaitTime := 1 * time.Second
	etime, err := asw.SetDetachRequestTime(generatedVolumeName, nodeName)
	if err != nil {
		t.Fatalf("SetDetachRequestTime failed. Expected: <no error> Actual: <%v>", err)
	}
	if etime >= maxWaitTime {
		t.Logf("SetDetachRequestTim Expected: <elapsed time %v is smaller than maxWaitTime %v> Actual <elapsed time is larger than maxWaitTime>", etime, maxWaitTime)
	}
	// Sleep and call SetDetachRequestTime again
	time.Sleep(maxWaitTime)
	etime, err = asw.SetDetachRequestTime(generatedVolumeName, nodeName)
	if err != nil {
		t.Fatalf("SetDetachRequestTime failed. Expected: <no error> Actual: <%v>", err)
	}
	if etime < maxWaitTime {
		t.Fatalf("SetDetachRequestTim Expected: <elapsed time %v is larger than maxWaitTime %v> Actual <elapsed time is smaller>", etime, maxWaitTime)
	}
}

func verifyAttachedVolume(
	t *testing.T,
	attachedVolumes []AttachedVolume,
	expectedVolumeName api.UniqueVolumeName,
	expectedVolumeSpecName string,
	expectedNodeName string,
	expectedMountedByNode,
	expectNonZeroDetachRequestedTime bool) {
	for _, attachedVolume := range attachedVolumes {
		if attachedVolume.VolumeName == expectedVolumeName &&
			attachedVolume.VolumeSpec.Name() == expectedVolumeSpecName &&
			attachedVolume.NodeName == expectedNodeName &&
			attachedVolume.MountedByNode == expectedMountedByNode &&
			attachedVolume.DetachRequestedTime.IsZero() == !expectNonZeroDetachRequestedTime {
			return
		}
	}

	t.Fatalf(
		"attachedVolumes (%v) should contain the volume/node combo %q/%q with MountedByNode=%v and NonZeroDetachRequestedTime=%v. It does not.",
		attachedVolumes,
		expectedVolumeName,
		expectedNodeName,
		expectedMountedByNode,
		expectNonZeroDetachRequestedTime)
}

func Test_GetAttachedVolumesForNode_Positive_NoVolumesOrNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	node := "random"

	// Act
	attachedVolumes := asw.GetAttachedVolumesForNode(node)

	// Assert
	if len(attachedVolumes) != 0 {
		t.Fatalf("len(attachedVolumes) Expected: <0> Actual: <%v>", len(attachedVolumes))
	}
}

func Test_GetAttachedVolumesForNode_Positive_OneVolumeOneNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := "node-name"
	devicePath := "fake/device/path"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName, devicePath)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	attachedVolumes := asw.GetAttachedVolumesForNode(nodeName)

	// Assert
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, string(volumeName), nodeName, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_GetAttachedVolumesForNode_Positive_TwoVolumeTwoNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volume1Name := api.UniqueVolumeName("volume1-name")
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(string(volume1Name), volume1Name)
	node1Name := "node1-name"
	devicePath := "fake/device/path"
	_, add1Err := asw.AddVolumeNode(volume1Spec, node1Name, devicePath)
	if add1Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add1Err)
	}
	volume2Name := api.UniqueVolumeName("volume2-name")
	volume2Spec := controllervolumetesting.GetTestVolumeSpec(string(volume2Name), volume2Name)
	node2Name := "node2-name"
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volume2Spec, node2Name, devicePath)
	if add2Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add2Err)
	}

	// Act
	attachedVolumes := asw.GetAttachedVolumesForNode(node2Name)

	// Assert
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName2, string(volume2Name), node2Name, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_GetAttachedVolumesForNode_Positive_OneVolumeTwoNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	node1Name := "node1-name"
	devicePath := "fake/device/path"
	generatedVolumeName1, add1Err := asw.AddVolumeNode(volumeSpec, node1Name, devicePath)
	if add1Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add1Err)
	}
	node2Name := "node2-name"
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volumeSpec, node2Name, devicePath)
	if add2Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add2Err)
	}

	if generatedVolumeName1 != generatedVolumeName2 {
		t.Fatalf(
			"Generated volume names for the same volume should be the same but they are not: %q and %q",
			generatedVolumeName1,
			generatedVolumeName2)
	}

	// Act
	attachedVolumes := asw.GetAttachedVolumesForNode(node1Name)

	// Assert
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, string(volumeName), node1Name, true /* expectedMountedByNode */, false /* expectNonZeroDetachRequestedTime */)
}
