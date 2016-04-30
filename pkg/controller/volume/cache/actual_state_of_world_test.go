/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/testing"
)

func Test_AddVolumeNode_Positive_NewVolumeNewNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)

	nodeName := "node-name"

	// Act
	generatedVolumeName, err := asw.AddVolumeNode(volumeSpec, nodeName)

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

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, nodeName, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_AddVolumeNode_Positive_ExistingVolumeNewNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	node1Name := "node1-name"
	node2Name := "node2-name"

	// Act
	generatedVolumeName1, add1Err := asw.AddVolumeNode(volumeSpec, node1Name)
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volumeSpec, node2Name)

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

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, volumeName, node1Name, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, volumeName, node2Name, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_AddVolumeNode_Positive_ExistingVolumeExistingNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"

	// Act
	generatedVolumeName1, add1Err := asw.AddVolumeNode(volumeSpec, nodeName)
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volumeSpec, nodeName)

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

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, volumeName, nodeName, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_DeleteVolumeNode_Positive_VolumeExistsNodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName)
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

func Test_DeleteVolumeNode_Positive_VolumeDoesntExistNodeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
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

func Test_DeleteVolumeNode_Positive_TwoNodesOneDeleted(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	node1Name := "node1-name"
	node2Name := "node2-name"
	generatedVolumeName1, add1Err := asw.AddVolumeNode(volumeSpec, node1Name)
	if add1Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add1Err)
	}
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volumeSpec, node2Name)
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

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, volumeName, node2Name, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_VolumeNodeExists_Positive_VolumeExistsNodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName)
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

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, nodeName, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_VolumeNodeExists_Positive_VolumeExistsNodeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	node1Name := "node1-name"
	node2Name := "node2-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, node1Name)
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

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, node1Name, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_VolumeNodeExists_Positive_VolumeAndNodeDontExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
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

func Test_GetAttachedVolumes_Positive_NoVolumesOrNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)

	// Act
	attachedVolumes := asw.GetAttachedVolumes()

	// Assert
	if len(attachedVolumes) != 0 {
		t.Fatalf("len(attachedVolumes) Expected: <0> Actual: <%v>", len(attachedVolumes))
	}
}

func Test_GetAttachedVolumes_Positive_OneVolumeOneNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	attachedVolumes := asw.GetAttachedVolumes()

	// Assert
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, nodeName, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_GetAttachedVolumes_Positive_TwoVolumeTwoNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volume1Name := "volume1-name"
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(volume1Name, volume1Name)
	node1Name := "node1-name"
	generatedVolumeName1, add1Err := asw.AddVolumeNode(volume1Spec, node1Name)
	if add1Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add1Err)
	}
	volume2Name := "volume2-name"
	volume2Spec := controllervolumetesting.GetTestVolumeSpec(volume2Name, volume2Name)
	node2Name := "node2-name"
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volume2Spec, node2Name)
	if add2Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add2Err)
	}

	// Act
	attachedVolumes := asw.GetAttachedVolumes()

	// Assert
	if len(attachedVolumes) != 2 {
		t.Fatalf("len(attachedVolumes) Expected: <2> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, volume1Name, node1Name, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName2, volume2Name, node2Name, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_GetAttachedVolumes_Positive_OneVolumeTwoNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	node1Name := "node1-name"
	generatedVolumeName1, add1Err := asw.AddVolumeNode(volumeSpec, node1Name)
	if add1Err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", add1Err)
	}
	node2Name := "node2-name"
	generatedVolumeName2, add2Err := asw.AddVolumeNode(volumeSpec, node2Name)
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

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, volumeName, node1Name, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName1, volumeName, node2Name, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_MarkVolumeNodeSafeToDetach_Positive_NotMarked(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act: do not mark -- test default value

	// Assert
	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, nodeName, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_MarkVolumeNodeSafeToDetach_Positive_Marked(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	markSafeToDetachErr := asw.MarkVolumeNodeSafeToDetach(generatedVolumeName, nodeName)

	// Assert
	if markSafeToDetachErr != nil {
		t.Fatalf("MarkVolumeNodeSafeToDetach failed. Expected <no error> Actual: <%v>", markSafeToDetachErr)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, nodeName, true /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_MarkVolumeNodeSafeToDetach_Positive_MarkedAddVolumeNodeReset(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	markSafeToDetachErr := asw.MarkVolumeNodeSafeToDetach(generatedVolumeName, nodeName)
	generatedVolumeName, addErr = asw.AddVolumeNode(volumeSpec, nodeName)

	// Assert
	if markSafeToDetachErr != nil {
		t.Fatalf("MarkVolumeNodeSafeToDetach failed. Expected <no error> Actual: <%v>", markSafeToDetachErr)
	}
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, nodeName, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_MarkVolumeNodeSafeToDetach_Positive_MarkedVerifyDetachRequestedTimePerserved(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}
	_, err := asw.MarkDesireToDetach(generatedVolumeName, nodeName)
	if err != nil {
		t.Fatalf("MarkDesireToDetach failed. Expected: <no error> Actual: <%v>", err)
	}
	expectedDetachRequestedTime := asw.GetAttachedVolumes()[0].DetachRequestedTime

	// Act
	markSafeToDetachErr := asw.MarkVolumeNodeSafeToDetach(generatedVolumeName, nodeName)

	// Assert
	if markSafeToDetachErr != nil {
		t.Fatalf("MarkVolumeNodeSafeToDetach failed. Expected <no error> Actual: <%v>", markSafeToDetachErr)
	}

	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, nodeName, true /* expectedSafeToDetach */, true /* expectNonZeroDetachRequestedTime */)
	if !expectedDetachRequestedTime.Equal(attachedVolumes[0].DetachRequestedTime) {
		t.Fatalf("DetachRequestedTime changed. Expected: <%v> Actual: <%v>", expectedDetachRequestedTime, attachedVolumes[0].DetachRequestedTime)
	}
}

func Test_MarkDesireToDetach_Positive_NotMarked(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act: do not mark -- test default value

	// Assert
	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, nodeName, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_MarkDesireToDetach_Positive_Marked(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	_, markDesireToDetachErr := asw.MarkDesireToDetach(generatedVolumeName, nodeName)

	// Assert
	if markDesireToDetachErr != nil {
		t.Fatalf("MarkDesireToDetach failed. Expected: <no error> Actual: <%v>", markDesireToDetachErr)
	}

	// Assert
	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, nodeName, false /* expectedSafeToDetach */, true /* expectNonZeroDetachRequestedTime */)
}

func Test_MarkDesireToDetach_Positive_MarkedAddVolumeNodeReset(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Act
	_, markDesireToDetachErr := asw.MarkDesireToDetach(generatedVolumeName, nodeName)
	generatedVolumeName, addErr = asw.AddVolumeNode(volumeSpec, nodeName)

	// Assert
	if markDesireToDetachErr != nil {
		t.Fatalf("MarkDesireToDetach failed. Expected: <no error> Actual: <%v>", markDesireToDetachErr)
	}
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}

	// Assert
	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, nodeName, false /* expectedSafeToDetach */, false /* expectNonZeroDetachRequestedTime */)
}

func Test_MarkDesireToDetach_Positive_MarkedVerifySafeToDetachPreserved(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	asw := NewActualStateOfWorld(volumePluginMgr)
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	generatedVolumeName, addErr := asw.AddVolumeNode(volumeSpec, nodeName)
	if addErr != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", addErr)
	}
	markSafeToDetachErr := asw.MarkVolumeNodeSafeToDetach(generatedVolumeName, nodeName)
	if markSafeToDetachErr != nil {
		t.Fatalf("MarkVolumeNodeSafeToDetach failed. Expected <no error> Actual: <%v>", markSafeToDetachErr)
	}

	// Act
	_, markDesireToDetachErr := asw.MarkDesireToDetach(generatedVolumeName, nodeName)

	// Assert
	if markDesireToDetachErr != nil {
		t.Fatalf("MarkDesireToDetach failed. Expected: <no error> Actual: <%v>", markDesireToDetachErr)
	}

	// Assert
	attachedVolumes := asw.GetAttachedVolumes()
	if len(attachedVolumes) != 1 {
		t.Fatalf("len(attachedVolumes) Expected: <1> Actual: <%v>", len(attachedVolumes))
	}

	verifyAttachedVolume(t, attachedVolumes, generatedVolumeName, volumeName, nodeName, true /* expectedSafeToDetach */, true /* expectNonZeroDetachRequestedTime */)
}

func verifyAttachedVolume(
	t *testing.T,
	attachedVolumes []AttachedVolume,
	expectedVolumeName,
	expectedVolumeSpecName,
	expectedNodeName string,
	expectedSafeToDetach,
	expectNonZeroDetachRequestedTime bool) {
	for _, attachedVolume := range attachedVolumes {
		if attachedVolume.VolumeName == expectedVolumeName &&
			attachedVolume.VolumeSpec.Name() == expectedVolumeSpecName &&
			attachedVolume.NodeName == expectedNodeName &&
			attachedVolume.SafeToDetach == expectedSafeToDetach &&
			attachedVolume.DetachRequestedTime.IsZero() == !expectNonZeroDetachRequestedTime {
			return
		}
	}

	t.Fatalf(
		"attachedVolumes (%v) should contain the volume/node combo %q/%q with SafeToDetach=%v and NonZeroDetachRequestedTime=%v. It does not.",
		attachedVolumes,
		expectedVolumeName,
		expectedNodeName,
		expectedSafeToDetach,
		expectNonZeroDetachRequestedTime)
}

// t.Logf("attachedVolumes: %v", asw.GetAttachedVolumes()) // TEMP
