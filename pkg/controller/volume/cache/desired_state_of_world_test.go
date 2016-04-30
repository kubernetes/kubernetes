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

func Test_AddNode_Positive_NewNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := "node-name"

	// Act
	dsw.AddNode(nodeName)

	// Assert
	nodeExists := dsw.NodeExists(nodeName)
	if !nodeExists {
		t.Fatalf("Added node %q does not exist, it should.", nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

func Test_AddNode_Positive_ExistingVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := "node-name"
	dsw.AddNode(nodeName)

	// Act
	dsw.AddNode(nodeName)

	// Assert
	nodeExists := dsw.NodeExists(nodeName)
	if !nodeExists {
		t.Fatalf("Added node %q does not exist, it should.", nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}
func Test_AddNode_Positive_ExistingNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := "node-name"

	// Act
	dsw.AddNode(nodeName)

	// Assert
	nodeExists := dsw.NodeExists(nodeName)
	if !nodeExists {
		t.Fatalf("Added node %q does not exist, it should.", nodeName)
	}

	// Act
	dsw.AddNode(nodeName)

	// Assert
	nodeExists = dsw.NodeExists(nodeName)
	if !nodeExists {
		t.Fatalf("Added node %q does not exist, it should.", nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

func Test_AddPod_Positive_NewPodNodeExistsVolumeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	podName := "pod-name"
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	dsw.AddNode(nodeName)
	volumeExists := dsw.VolumeExists(volumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName,
			nodeName)
	}

	// Act
	generatedVolumeName, podErr := dsw.AddPod(podName, volumeSpec, nodeName)

	// Assert
	if podErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podErr)
	}

	volumeExists = dsw.VolumeExists(generatedVolumeName, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Added pod %q to volume %q/node %q. Volume does not exist, it should.",
			podName,
			generatedVolumeName,
			nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 1 {
		t.Fatalf("len(volumesToAttach) Expected: <1> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, volumeName)
}

func Test_AddPod_Positive_NewPodNodeExistsVolumeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	pod1Name := "pod1-name"
	pod2Name := "pod2-name"
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	dsw.AddNode(nodeName)
	volumeExists := dsw.VolumeExists(volumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName,
			nodeName)
	}

	// Act
	generatedVolumeName, podErr := dsw.AddPod(pod1Name, volumeSpec, nodeName)

	// Assert
	if podErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod1Name,
			podErr)
	}

	volumeExists = dsw.VolumeExists(generatedVolumeName, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Added pod %q to volume %q/node %q. Volume does not exist, it should.",
			pod1Name,
			generatedVolumeName,
			nodeName)
	}

	// Act
	generatedVolumeName, podErr = dsw.AddPod(pod2Name, volumeSpec, nodeName)

	// Assert
	if podErr != nil {
		t.Fatalf("AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod2Name,
			podErr)
	}

	volumeExists = dsw.VolumeExists(generatedVolumeName, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Added pod %q to volume %q/node %q. Volume does not exist, it should.",
			pod1Name,
			generatedVolumeName,
			nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 1 {
		t.Fatalf("len(volumesToAttach) Expected: <1> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, volumeName)
}

func Test_AddPod_Positive_PodExistsNodeExistsVolumeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	podName := "pod-name"
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	dsw.AddNode(nodeName)
	volumeExists := dsw.VolumeExists(volumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName,
			nodeName)
	}

	// Act
	generatedVolumeName, podErr := dsw.AddPod(podName, volumeSpec, nodeName)

	// Assert
	if podErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			podName,
			podErr)
	}

	volumeExists = dsw.VolumeExists(generatedVolumeName, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Added pod %q to volume %q/node %q. Volume does not exist, it should.",
			podName,
			generatedVolumeName,
			nodeName)
	}

	// Act
	generatedVolumeName, podErr = dsw.AddPod(podName, volumeSpec, nodeName)

	// Assert
	if podErr != nil {
		t.Fatalf("AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			podName,
			podErr)
	}

	volumeExists = dsw.VolumeExists(generatedVolumeName, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Added pod %q to volume %q/node %q. Volume does not exist, it should.",
			podName,
			generatedVolumeName,
			nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 1 {
		t.Fatalf("len(volumesToAttach) Expected: <1> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, volumeName)
}

func Test_AddPod_Negative_NewPodNodeDoesntExistVolumeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	podName := "pod-name"
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	volumeExists := dsw.VolumeExists(volumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName,
			nodeName)
	}

	// Act
	_, podErr := dsw.AddPod(podName, volumeSpec, nodeName)

	// Assert
	if podErr == nil {
		t.Fatalf("AddPod did not fail. Expected: <\"failed to add pod...no node with that name exists in the list of managed nodes\"> Actual: <no error>")
	}

	volumeExists = dsw.VolumeExists(volumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName,
			nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

func Test_DeleteNode_Positive_NodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := "node-name"
	dsw.AddNode(nodeName)

	// Act
	err := dsw.DeleteNode(nodeName)

	// Assert
	if err != nil {
		t.Fatalf("DeleteNode failed. Expected: <no error> Actual: <%v>", err)
	}

	nodeExists := dsw.NodeExists(nodeName)
	if nodeExists {
		t.Fatalf("Deleted node %q still exists, it should not.", nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

func Test_DeleteNode_Positive_NodeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	notAddedNodeName := "node-not-added-name"

	// Act
	err := dsw.DeleteNode(notAddedNodeName)

	// Assert
	if err != nil {
		t.Fatalf("DeleteNode failed. Expected: <no error> Actual: <%v>", err)
	}

	nodeExists := dsw.NodeExists(notAddedNodeName)
	if nodeExists {
		t.Fatalf("Deleted node %q still exists, it should not.", notAddedNodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

func Test_DeleteNode_Negative_NodeExistsHasChildVolumes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := "node-name"
	dsw.AddNode(nodeName)
	podName := "pod-name"
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	generatedVolumeName, podAddErr := dsw.AddPod(podName, volumeSpec, nodeName)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			podName,
			podAddErr)
	}

	// Act
	err := dsw.DeleteNode(nodeName)

	// Assert
	if err == nil {
		t.Fatalf("DeleteNode did not fail. Expected: <\"\"> Actual: <no error>")
	}

	nodeExists := dsw.NodeExists(nodeName)
	if !nodeExists {
		t.Fatalf("Node %q no longer exists, it should.", nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 1 {
		t.Fatalf("len(volumesToAttach) Expected: <1> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, volumeName)
}

func Test_DeletePod_Positive_PodExistsNodeExistsVolumeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	podName := "pod-name"
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	dsw.AddNode(nodeName)
	generatedVolumeName, podAddErr := dsw.AddPod(podName, volumeSpec, nodeName)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			podName,
			podAddErr)
	}
	volumeExists := dsw.VolumeExists(generatedVolumeName, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Added pod %q to volume %q/node %q. Volume does not exist, it should.",
			podName,
			generatedVolumeName,
			nodeName)
	}

	// Act
	dsw.DeletePod(podName, generatedVolumeName, nodeName)

	// Assert
	volumeExists = dsw.VolumeExists(generatedVolumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Deleted pod %q from volume %q/node %q. Volume should also be deleted but it still exists.",
			podName,
			generatedVolumeName,
			nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

func Test_DeletePod_Positive_2PodsExistNodeExistsVolumesExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	pod1Name := "pod1-name"
	pod2Name := "pod2-name"
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	dsw.AddNode(nodeName)
	generatedVolumeName1, pod1AddErr := dsw.AddPod(pod1Name, volumeSpec, nodeName)
	if pod1AddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod1Name,
			pod1AddErr)
	}
	generatedVolumeName2, pod2AddErr := dsw.AddPod(pod2Name, volumeSpec, nodeName)
	if pod2AddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod2Name,
			pod2AddErr)
	}
	if generatedVolumeName1 != generatedVolumeName2 {
		t.Fatalf(
			"Generated volume names for the same volume should be the same but they are not: %q and %q",
			generatedVolumeName1,
			generatedVolumeName2)
	}
	volumeExists := dsw.VolumeExists(generatedVolumeName1, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Volume %q does not exist under node %q, it should.",
			generatedVolumeName1,
			nodeName)
	}

	// Act
	dsw.DeletePod(pod1Name, generatedVolumeName1, nodeName)

	// Assert
	volumeExists = dsw.VolumeExists(generatedVolumeName1, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Volume %q under node %q should still exist, but it does not.",
			generatedVolumeName1,
			nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 1 {
		t.Fatalf("len(volumesToAttach) Expected: <1> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName1, volumeName)
}

func Test_DeletePod_Positive_PodDoesNotExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	pod1Name := "pod1-name"
	pod2Name := "pod2-name"
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	nodeName := "node-name"
	dsw.AddNode(nodeName)
	generatedVolumeName, pod1AddErr := dsw.AddPod(pod1Name, volumeSpec, nodeName)
	if pod1AddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod1Name,
			pod1AddErr)
	}
	volumeExists := dsw.VolumeExists(generatedVolumeName, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Added pod %q to volume %q/node %q. Volume does not exist, it should.",
			pod1Name,
			generatedVolumeName,
			nodeName)
	}

	// Act
	dsw.DeletePod(pod2Name, generatedVolumeName, nodeName)

	// Assert
	volumeExists = dsw.VolumeExists(generatedVolumeName, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Volume %q/node %q does not exist, it should.",
			generatedVolumeName,
			nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 1 {
		t.Fatalf("len(volumesToAttach) Expected: <1> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, volumeName)
}

func Test_DeletePod_Positive_NodeDoesNotExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	podName := "pod-name"
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	node1Name := "node1-name"
	dsw.AddNode(node1Name)
	generatedVolumeName, podAddErr := dsw.AddPod(podName, volumeSpec, node1Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			podName,
			podAddErr)
	}
	volumeExists := dsw.VolumeExists(generatedVolumeName, node1Name)
	if !volumeExists {
		t.Fatalf(
			"Added pod %q to volume %q/node %q. Volume does not exist, it should.",
			podName,
			generatedVolumeName,
			node1Name)
	}
	node2Name := "node2-name"

	// Act
	dsw.DeletePod(podName, generatedVolumeName, node2Name)

	// Assert
	volumeExists = dsw.VolumeExists(generatedVolumeName, node1Name)
	if !volumeExists {
		t.Fatalf(
			"Volume %q/node %q does not exist, it should.",
			generatedVolumeName,
			node1Name)
	}
	volumeExists = dsw.VolumeExists(generatedVolumeName, node2Name)
	if volumeExists {
		t.Fatalf(
			"node %q exists, it should not.",
			node2Name)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 1 {
		t.Fatalf("len(volumesToAttach) Expected: <1> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, node1Name, generatedVolumeName, volumeName)
}

func Test_DeletePod_Positive_VolumeDoesNotExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	podName := "pod-name"
	volume1Name := "volume1-name"
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(volume1Name, volume1Name)
	nodeName := "node-name"
	dsw.AddNode(nodeName)
	generatedVolume1Name, podAddErr := dsw.AddPod(podName, volume1Spec, nodeName)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			podName,
			podAddErr)
	}
	volumeExists := dsw.VolumeExists(generatedVolume1Name, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Added pod %q to volume %q/node %q. Volume does not exist, it should.",
			podName,
			generatedVolume1Name,
			nodeName)
	}
	volume2Name := "volume2-name"

	// Act
	dsw.DeletePod(podName, volume2Name, nodeName)

	// Assert
	volumeExists = dsw.VolumeExists(generatedVolume1Name, nodeName)
	if !volumeExists {
		t.Fatalf(
			"Volume %q/node %q does not exist, it should.",
			generatedVolume1Name,
			nodeName)
	}
	volumeExists = dsw.VolumeExists(volume2Name, nodeName)
	if volumeExists {
		t.Fatalf(
			"volume %q exists, it should not.",
			volume2Name)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 1 {
		t.Fatalf("len(volumesToAttach) Expected: <1> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolume1Name, volume1Name)
}

func Test_NodeExists_Positive_NodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	notAddedNodeName := "node-not-added-name"

	// Act
	notAddedNodeExists := dsw.NodeExists(notAddedNodeName)

	// Assert
	if notAddedNodeExists {
		t.Fatalf("Node %q exists, it should not.", notAddedNodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

func Test_NodeExists_Positive_NodeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := "node-name"
	dsw.AddNode(nodeName)

	// Act
	nodeExists := dsw.NodeExists(nodeName)

	// Assert
	if !nodeExists {
		t.Fatalf("Node %q does not exist, it should.", nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

func Test_VolumeExists_Positive_VolumeExistsNodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := "node-name"
	dsw.AddNode(nodeName)
	podName := "pod-name"
	volumeName := "volume-name"
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(volumeName, volumeName)
	generatedVolumeName, _ := dsw.AddPod(podName, volumeSpec, nodeName)

	// Act
	volumeExists := dsw.VolumeExists(generatedVolumeName, nodeName)

	// Assert
	if !volumeExists {
		t.Fatalf("Volume %q does not exist, it should.", generatedVolumeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 1 {
		t.Fatalf("len(volumesToAttach) Expected: <1> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, volumeName)
}

func Test_VolumeExists_Positive_VolumeDoesntExistNodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := "node-name"
	dsw.AddNode(nodeName)
	podName := "pod-name"
	volume1Name := "volume1-name"
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(volume1Name, volume1Name)
	generatedVolume1Name, podAddErr := dsw.AddPod(podName, volume1Spec, nodeName)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			podName,
			podAddErr)
	}
	volume2Name := "volume2-name"

	// Act
	volumeExists := dsw.VolumeExists(volume2Name, nodeName)

	// Assert
	if volumeExists {
		t.Fatalf("Volume %q exists, it should not.", volume2Name)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 1 {
		t.Fatalf("len(volumesToAttach) Expected: <1> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolume1Name, volume1Name)
}

func Test_VolumeExists_Positive_VolumeDoesntExistNodeDoesntExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := "node-name"
	volumeName := "volume-name"

	// Act
	volumeExists := dsw.VolumeExists(volumeName, nodeName)

	// Assert
	if volumeExists {
		t.Fatalf("Volume %q exists, it should not.", volumeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

func Test_GetVolumesToAttach_Positive_NoNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)

	// Act
	volumesToAttach := dsw.GetVolumesToAttach()

	// Assert
	if len(volumesToAttach) > 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

func Test_GetVolumesToAttach_Positive_TwoNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	node1Name := "node1-name"
	node2Name := "node2-name"
	dsw.AddNode(node1Name)
	dsw.AddNode(node2Name)

	// Act
	volumesToAttach := dsw.GetVolumesToAttach()

	// Assert
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

func Test_GetVolumesToAttach_Positive_TwoNodesOneVolumeEach(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	node1Name := "node1-name"
	pod1Name := "pod1-name"
	volume1Name := "volume1-name"
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(volume1Name, volume1Name)
	dsw.AddNode(node1Name)
	generatedVolume1Name, podAddErr := dsw.AddPod(pod1Name, volume1Spec, node1Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod1Name,
			podAddErr)
	}
	node2Name := "node2-name"
	pod2Name := "pod2-name"
	volume2Name := "volume2-name"
	volume2Spec := controllervolumetesting.GetTestVolumeSpec(volume2Name, volume2Name)
	dsw.AddNode(node2Name)
	generatedVolume2Name, podAddErr := dsw.AddPod(pod2Name, volume2Spec, node2Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod2Name,
			podAddErr)
	}

	// Act
	volumesToAttach := dsw.GetVolumesToAttach()

	// Assert
	if len(volumesToAttach) != 2 {
		t.Fatalf("len(volumesToAttach) Expected: <2> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, node1Name, generatedVolume1Name, volume1Name)
	verifyVolumeToAttach(t, volumesToAttach, node2Name, generatedVolume2Name, volume2Name)
}

func Test_GetVolumesToAttach_Positive_TwoNodesOneVolumeEachExtraPod(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	node1Name := "node1-name"
	pod1Name := "pod1-name"
	volume1Name := "volume1-name"
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(volume1Name, volume1Name)
	dsw.AddNode(node1Name)
	generatedVolume1Name, podAddErr := dsw.AddPod(pod1Name, volume1Spec, node1Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod1Name,
			podAddErr)
	}
	node2Name := "node2-name"
	pod2Name := "pod2-name"
	volume2Name := "volume2-name"
	volume2Spec := controllervolumetesting.GetTestVolumeSpec(volume2Name, volume2Name)
	dsw.AddNode(node2Name)
	generatedVolume2Name, podAddErr := dsw.AddPod(pod2Name, volume2Spec, node2Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod2Name,
			podAddErr)
	}
	pod3Name := "pod3-name"
	dsw.AddPod(pod3Name, volume2Spec, node2Name)
	_, podAddErr = dsw.AddPod(pod3Name, volume2Spec, node2Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod3Name,
			podAddErr)
	}

	// Act
	volumesToAttach := dsw.GetVolumesToAttach()

	// Assert
	if len(volumesToAttach) != 2 {
		t.Fatalf("len(volumesToAttach) Expected: <2> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, node1Name, generatedVolume1Name, volume1Name)
	verifyVolumeToAttach(t, volumesToAttach, node2Name, generatedVolume2Name, volume2Name)
}

func Test_GetVolumesToAttach_Positive_TwoNodesThreeVolumes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := controllervolumetesting.GetTestVolumePluginMgr((t))
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	node1Name := "node1-name"
	pod1Name := "pod1-name"
	volume1Name := "volume1-name"
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(volume1Name, volume1Name)
	dsw.AddNode(node1Name)
	generatedVolume1Name, podAddErr := dsw.AddPod(pod1Name, volume1Spec, node1Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod1Name,
			podAddErr)
	}
	node2Name := "node2-name"
	pod2aName := "pod2a-name"
	volume2Name := "volume2-name"
	volume2Spec := controllervolumetesting.GetTestVolumeSpec(volume2Name, volume2Name)
	dsw.AddNode(node2Name)
	generatedVolume2Name1, podAddErr := dsw.AddPod(pod2aName, volume2Spec, node2Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod2aName,
			podAddErr)
	}
	pod2bName := "pod2b-name"
	generatedVolume2Name2, podAddErr := dsw.AddPod(pod2bName, volume2Spec, node2Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod2bName,
			podAddErr)
	}
	if generatedVolume2Name1 != generatedVolume2Name2 {
		t.Fatalf(
			"Generated volume names for the same volume should be the same but they are not: %q and %q",
			generatedVolume2Name1,
			generatedVolume2Name2)
	}
	pod3Name := "pod3-name"
	volume3Name := "volume3-name"
	volume3Spec := controllervolumetesting.GetTestVolumeSpec(volume3Name, volume3Name)
	generatedVolume3Name, podAddErr := dsw.AddPod(pod3Name, volume3Spec, node1Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod3Name,
			podAddErr)
	}

	// Act
	volumesToAttach := dsw.GetVolumesToAttach()

	// Assert
	if len(volumesToAttach) != 3 {
		t.Fatalf("len(volumesToAttach) Expected: <3> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, node1Name, generatedVolume1Name, volume1Name)
	verifyVolumeToAttach(t, volumesToAttach, node2Name, generatedVolume2Name1, volume2Name)
	verifyVolumeToAttach(t, volumesToAttach, node1Name, generatedVolume3Name, volume3Name)
}

func verifyVolumeToAttach(
	t *testing.T,
	volumesToAttach []VolumeToAttach,
	expectedNodeName,
	expectedVolumeName,
	expectedVolumeSpecName string) {
	for _, volumeToAttach := range volumesToAttach {
		if volumeToAttach.NodeName == expectedNodeName &&
			volumeToAttach.VolumeName == expectedVolumeName &&
			volumeToAttach.VolumeSpec.Name() == expectedVolumeSpecName {
			return
		}
	}

	t.Fatalf("volumesToAttach (%v) should contain %q/%q. It does not.", volumesToAttach, expectedVolumeName, expectedNodeName)
}
