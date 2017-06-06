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

	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

// Calls AddNode() once.
// Verifies node exists, and zero volumes to attach.
func Test_AddNode_Positive_NewNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := k8stypes.NodeName("node-name")

	// Act
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)

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

// Calls AddNode() once.
// Verifies node exists.
// Calls AddNode() again with the same node.
// Verifies node exists, and zero volumes to attach.
func Test_AddNode_Positive_ExistingNode(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := k8stypes.NodeName("node-name")

	// Act
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)

	// Assert
	nodeExists := dsw.NodeExists(nodeName)
	if !nodeExists {
		t.Fatalf("Added node %q does not exist, it should.", nodeName)
	}

	// Act
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)

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

// Populates data struct with a single node no volume.
// Calls AddPod() with the same node and new pod/volume.
// Verifies node/volume exists, and 1 volumes to attach.
func Test_AddPod_Positive_NewPodNodeExistsVolumeDoesntExist(t *testing.T) {
	// Arrange
	podName := "pod-uid"
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)
	volumeExists := dsw.VolumeExists(volumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName,
			nodeName)
	}

	// Act
	generatedVolumeName, podErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)

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

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, string(volumeName))
}

// Populates data struct with a single node no volume.
// Calls AddPod() with the same node and new pod/volume.
// Verifies node/volume exists.
// Calls AddPod() with the same node and volume different pod.
// Verifies the same node/volume exists, and 1 volumes to attach.
func Test_AddPod_Positive_NewPodNodeExistsVolumeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	pod1Name := "pod1-uid"
	pod2Name := "pod2-uid"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)
	volumeExists := dsw.VolumeExists(volumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName,
			nodeName)
	}

	// Act
	generatedVolumeName, podErr := dsw.AddPod(types.UniquePodName(pod1Name), controllervolumetesting.NewPod(pod1Name, pod1Name), volumeSpec, nodeName)

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
	generatedVolumeName, podErr = dsw.AddPod(types.UniquePodName(pod2Name), controllervolumetesting.NewPod(pod2Name, pod2Name), volumeSpec, nodeName)

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

	podScheduled := dsw.GetPodToAdd()
	if len(podScheduled) != 2 {
		t.Fatalf("len(podScheduled) Expected: <2> Actual: <%v>", len(podScheduled))
	}
	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, string(volumeName))
}

// Populates data struct with a single node no volume.
// Calls AddPod() with the same node and new pod/volume.
// Verifies node/volume exists.
// Calls AddPod() with the same node, volume, and pod.
// Verifies the same node/volume exists, and 1 volumes to attach.
func Test_AddPod_Positive_PodExistsNodeExistsVolumeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	podName := "pod-uid"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)
	volumeExists := dsw.VolumeExists(volumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName,
			nodeName)
	}

	// Act
	generatedVolumeName, podErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)

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
	generatedVolumeName, podErr = dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)

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

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, string(volumeName))
}

// Calls AddPod() with new pod/volume/node on empty data struct.
// Verifies call fails because node does not exist.
func Test_AddPod_Negative_NewPodNodeDoesntExistVolumeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	podName := "pod-uid"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")
	volumeExists := dsw.VolumeExists(volumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName,
			nodeName)
	}

	// Act
	_, podErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)

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

// Populates data struct with a single node.
// Calls DeleteNode() to delete the node.
// Verifies node no longer exists, and zero volumes to attach.
func Test_DeleteNode_Positive_NodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)

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

// Calls DeleteNode() to delete node on empty data struct.
// Verifies no error is returned, and zero volumes to attach.
func Test_DeleteNode_Positive_NodeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	notAddedNodeName := k8stypes.NodeName("node-not-added-name")

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

// Populates data struct with new pod/volume/node.
// Calls DeleteNode() to delete the node.
// Verifies call fails because node still contains child volumes.
func Test_DeleteNode_Negative_NodeExistsHasChildVolumes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)
	podName := "pod-uid"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	generatedVolumeName, podAddErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)
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
		t.Fatalf("DeleteNode did not fail. Expected: <\"failed to delete node...the node still contains volumes in its list of volumes to attach\"> Actual: <no error>")
	}

	nodeExists := dsw.NodeExists(nodeName)
	if !nodeExists {
		t.Fatalf("Node %q no longer exists, it should.", nodeName)
	}

	volumesToAttach := dsw.GetVolumesToAttach()
	if len(volumesToAttach) != 1 {
		t.Fatalf("len(volumesToAttach) Expected: <1> Actual: <%v>", len(volumesToAttach))
	}

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, string(volumeName))
}

// Populates data struct with new pod/volume/node.
// Calls DeleteNode() to delete the pod/volume/node.
// Verifies volume no longer exists, and zero volumes to attach.
func Test_DeletePod_Positive_PodExistsNodeExistsVolumeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	podName := "pod-uid"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)
	generatedVolumeName, podAddErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)
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
	dsw.DeletePod(types.UniquePodName(podName), generatedVolumeName, nodeName)

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

// Populates data struct with pod1/volume/node and pod2/volume/node.
// Calls DeleteNode() to delete the pod1/volume/node.
// Verifies volume still exists, and one volumes to attach.
func Test_DeletePod_Positive_2PodsExistNodeExistsVolumesExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	pod1Name := "pod1-uid"
	pod2Name := "pod2-uid"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)
	generatedVolumeName1, pod1AddErr := dsw.AddPod(types.UniquePodName(pod1Name), controllervolumetesting.NewPod(pod1Name, pod1Name), volumeSpec, nodeName)
	if pod1AddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod1Name,
			pod1AddErr)
	}
	generatedVolumeName2, pod2AddErr := dsw.AddPod(types.UniquePodName(pod2Name), controllervolumetesting.NewPod(pod2Name, pod2Name), volumeSpec, nodeName)
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
	dsw.DeletePod(types.UniquePodName(pod1Name), generatedVolumeName1, nodeName)

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

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName1, string(volumeName))
}

// Populates data struct with pod1/volume/node.
// Calls DeleteNode() to delete the pod2/volume/node.
// Verifies volume still exists, and one volumes to attach.
func Test_DeletePod_Positive_PodDoesNotExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	pod1Name := "pod1-uid"
	pod2Name := "pod2-uid"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)
	generatedVolumeName, pod1AddErr := dsw.AddPod(types.UniquePodName(pod1Name), controllervolumetesting.NewPod(pod1Name, pod1Name), volumeSpec, nodeName)
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
	dsw.DeletePod(types.UniquePodName(pod2Name), generatedVolumeName, nodeName)

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

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, string(volumeName))
}

// Populates data struct with pod/volume/node1.
// Calls DeleteNode() to delete the pod/volume/node2.
// Verifies volume still exists, and one volumes to attach.
func Test_DeletePod_Positive_NodeDoesNotExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	podName := "pod-uid"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	node1Name := k8stypes.NodeName("node1-name")
	dsw.AddNode(node1Name, false /*keepTerminatedPodVolumes*/)
	generatedVolumeName, podAddErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, node1Name)
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
	node2Name := k8stypes.NodeName("node2-name")

	// Act
	dsw.DeletePod(types.UniquePodName(podName), generatedVolumeName, node2Name)

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

	verifyVolumeToAttach(t, volumesToAttach, node1Name, generatedVolumeName, string(volumeName))
}

// Populates data struct with pod/volume1/node.
// Calls DeleteNode() to delete the pod/volume2/node.
// Verifies volume still exists, and one volumes to attach.
func Test_DeletePod_Positive_VolumeDoesNotExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	podName := "pod-uid"
	volume1Name := v1.UniqueVolumeName("volume1-name")
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(string(volume1Name), volume1Name)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)
	generatedVolume1Name, podAddErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volume1Spec, nodeName)
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
	volume2Name := v1.UniqueVolumeName("volume2-name")

	// Act
	dsw.DeletePod(types.UniquePodName(podName), volume2Name, nodeName)

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

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolume1Name, string(volume1Name))
}

// Calls NodeExists() on random node.
// Verifies node does not exist, and no volumes to attach.
func Test_NodeExists_Positive_NodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	notAddedNodeName := k8stypes.NodeName("node-not-added-name")

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

// Populates data struct with a single node.
// Calls NodeExists() on that node.
// Verifies node exists, and no volumes to attach.
func Test_NodeExists_Positive_NodeDoesntExist(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)

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

// Populates data struct with new pod/volume/node.
// Calls VolumeExists() on that volume/node.
// Verifies volume/node exists, and one volume to attach.
func Test_VolumeExists_Positive_VolumeExistsNodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)
	podName := "pod-uid"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	generatedVolumeName, _ := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)

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

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolumeName, string(volumeName))
}

// Populates data struct with new pod/volume1/node.
// Calls VolumeExists() on that volume2/node.
// Verifies volume2/node does not exist, and one volume to attach.
func Test_VolumeExists_Positive_VolumeDoesntExistNodeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName, false /*keepTerminatedPodVolumes*/)
	podName := "pod-uid"
	volume1Name := v1.UniqueVolumeName("volume1-name")
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(string(volume1Name), volume1Name)
	generatedVolume1Name, podAddErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volume1Spec, nodeName)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			podName,
			podAddErr)
	}
	volume2Name := v1.UniqueVolumeName("volume2-name")

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

	verifyVolumeToAttach(t, volumesToAttach, nodeName, generatedVolume1Name, string(volume1Name))
}

// Calls VolumeExists() on some volume/node.
// Verifies volume/node do not exist, and zero volumes to attach.
func Test_VolumeExists_Positive_VolumeDoesntExistNodeDoesntExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	nodeName := k8stypes.NodeName("node-name")
	volumeName := v1.UniqueVolumeName("volume-name")

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

// Calls GetVolumesToAttach()
// Verifies zero volumes to attach.
func Test_GetVolumesToAttach_Positive_NoNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)

	// Act
	volumesToAttach := dsw.GetVolumesToAttach()

	// Assert
	if len(volumesToAttach) > 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

// Populates data struct with two nodes.
// Calls GetVolumesToAttach()
// Verifies zero volumes to attach.
func Test_GetVolumesToAttach_Positive_TwoNodes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	node1Name := k8stypes.NodeName("node1-name")
	node2Name := k8stypes.NodeName("node2-name")
	dsw.AddNode(node1Name, false /*keepTerminatedPodVolumes*/)
	dsw.AddNode(node2Name, false /*keepTerminatedPodVolumes*/)

	// Act
	volumesToAttach := dsw.GetVolumesToAttach()

	// Assert
	if len(volumesToAttach) != 0 {
		t.Fatalf("len(volumesToAttach) Expected: <0> Actual: <%v>", len(volumesToAttach))
	}
}

// Populates data struct with two nodes with one volume/pod each.
// Calls GetVolumesToAttach()
// Verifies two volumes to attach.
func Test_GetVolumesToAttach_Positive_TwoNodesOneVolumeEach(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	node1Name := k8stypes.NodeName("node1-name")
	pod1Name := "pod1-uid"
	volume1Name := v1.UniqueVolumeName("volume1-name")
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(string(volume1Name), volume1Name)
	dsw.AddNode(node1Name, false /*keepTerminatedPodVolumes*/)
	generatedVolume1Name, podAddErr := dsw.AddPod(types.UniquePodName(pod1Name), controllervolumetesting.NewPod(pod1Name, pod1Name), volume1Spec, node1Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod1Name,
			podAddErr)
	}
	node2Name := k8stypes.NodeName("node2-name")
	pod2Name := "pod2-uid"
	volume2Name := v1.UniqueVolumeName("volume2-name")
	volume2Spec := controllervolumetesting.GetTestVolumeSpec(string(volume2Name), volume2Name)
	dsw.AddNode(node2Name, false /*keepTerminatedPodVolumes*/)
	generatedVolume2Name, podAddErr := dsw.AddPod(types.UniquePodName(pod2Name), controllervolumetesting.NewPod(pod2Name, pod2Name), volume2Spec, node2Name)
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

	verifyVolumeToAttach(t, volumesToAttach, node1Name, generatedVolume1Name, string(volume1Name))
	verifyVolumeToAttach(t, volumesToAttach, node2Name, generatedVolume2Name, string(volume2Name))
}

// Populates data struct with two nodes with one volume/pod each and an extra
// pod for the second node/volume pair.
// Calls GetVolumesToAttach()
// Verifies two volumes to attach.
func Test_GetVolumesToAttach_Positive_TwoNodesOneVolumeEachExtraPod(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	node1Name := k8stypes.NodeName("node1-name")
	pod1Name := "pod1-uid"
	volume1Name := v1.UniqueVolumeName("volume1-name")
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(string(volume1Name), volume1Name)
	dsw.AddNode(node1Name, false /*keepTerminatedPodVolumes*/)
	generatedVolume1Name, podAddErr := dsw.AddPod(types.UniquePodName(pod1Name), controllervolumetesting.NewPod(pod1Name, pod1Name), volume1Spec, node1Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod1Name,
			podAddErr)
	}
	node2Name := k8stypes.NodeName("node2-name")
	pod2Name := "pod2-uid"
	volume2Name := v1.UniqueVolumeName("volume2-name")
	volume2Spec := controllervolumetesting.GetTestVolumeSpec(string(volume2Name), volume2Name)
	dsw.AddNode(node2Name, false /*keepTerminatedPodVolumes*/)
	generatedVolume2Name, podAddErr := dsw.AddPod(types.UniquePodName(pod2Name), controllervolumetesting.NewPod(pod2Name, pod2Name), volume2Spec, node2Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod2Name,
			podAddErr)
	}
	pod3Name := "pod3-uid"
	dsw.AddPod(types.UniquePodName(pod3Name), controllervolumetesting.NewPod(pod3Name, pod3Name), volume2Spec, node2Name)
	_, podAddErr = dsw.AddPod(types.UniquePodName(pod3Name), controllervolumetesting.NewPod(pod3Name, pod3Name), volume2Spec, node2Name)
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

	verifyVolumeToAttach(t, volumesToAttach, node1Name, generatedVolume1Name, string(volume1Name))
	verifyVolumeToAttach(t, volumesToAttach, node2Name, generatedVolume2Name, string(volume2Name))
}

// Populates data struct with two nodes with one volume/pod on one node and two
// volume/pod pairs on the other node.
// Calls GetVolumesToAttach()
// Verifies three volumes to attach.
func Test_GetVolumesToAttach_Positive_TwoNodesThreeVolumes(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := NewDesiredStateOfWorld(volumePluginMgr)
	node1Name := k8stypes.NodeName("node1-name")
	pod1Name := "pod1-uid"
	volume1Name := v1.UniqueVolumeName("volume1-name")
	volume1Spec := controllervolumetesting.GetTestVolumeSpec(string(volume1Name), volume1Name)
	dsw.AddNode(node1Name, false /*keepTerminatedPodVolumes*/)
	generatedVolume1Name, podAddErr := dsw.AddPod(types.UniquePodName(pod1Name), controllervolumetesting.NewPod(pod1Name, pod1Name), volume1Spec, node1Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod1Name,
			podAddErr)
	}
	node2Name := k8stypes.NodeName("node2-name")
	pod2aName := "pod2a-name"
	volume2Name := v1.UniqueVolumeName("volume2-name")
	volume2Spec := controllervolumetesting.GetTestVolumeSpec(string(volume2Name), volume2Name)
	dsw.AddNode(node2Name, false /*keepTerminatedPodVolumes*/)
	generatedVolume2Name1, podAddErr := dsw.AddPod(types.UniquePodName(pod2aName), controllervolumetesting.NewPod(pod2aName, pod2aName), volume2Spec, node2Name)
	if podAddErr != nil {
		t.Fatalf(
			"AddPod failed for pod %q. Expected: <no error> Actual: <%v>",
			pod2aName,
			podAddErr)
	}
	pod2bName := "pod2b-name"
	generatedVolume2Name2, podAddErr := dsw.AddPod(types.UniquePodName(pod2bName), controllervolumetesting.NewPod(pod2bName, pod2bName), volume2Spec, node2Name)
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
	pod3Name := "pod3-uid"
	volume3Name := v1.UniqueVolumeName("volume3-name")
	volume3Spec := controllervolumetesting.GetTestVolumeSpec(string(volume3Name), volume3Name)
	generatedVolume3Name, podAddErr := dsw.AddPod(types.UniquePodName(pod3Name), controllervolumetesting.NewPod(pod3Name, pod3Name), volume3Spec, node1Name)
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

	verifyVolumeToAttach(t, volumesToAttach, node1Name, generatedVolume1Name, string(volume1Name))
	verifyVolumeToAttach(t, volumesToAttach, node2Name, generatedVolume2Name1, string(volume2Name))
	verifyVolumeToAttach(t, volumesToAttach, node1Name, generatedVolume3Name, string(volume3Name))
}

func verifyVolumeToAttach(
	t *testing.T,
	volumesToAttach []VolumeToAttach,
	expectedNodeName k8stypes.NodeName,
	expectedVolumeName v1.UniqueVolumeName,
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
