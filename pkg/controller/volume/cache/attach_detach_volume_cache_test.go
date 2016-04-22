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

import "testing"

func Test_AddVolume_Positive_NewVolume(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"

	// Act
	vc.AddVolume(volumeName)

	// Assert
	volumeExists := vc.VolumeExists(volumeName)
	if !volumeExists {
		t.Fatalf("Added volume %q does not exist, it should.", volumeName)
	}
}

func Test_AddVolume_Positive_ExistingVolume(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	vc.AddVolume(volumeName)

	// Act
	vc.AddVolume(volumeName)

	// Assert
	volumeExists := vc.VolumeExists(volumeName)
	if !volumeExists {
		t.Fatalf("Added volume %q does not exist, it should.", volumeName)
	}
}

func Test_AddNode_Positive_NewNodeVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	vc.AddVolume(volumeName)

	// Act
	nodeErr := vc.AddNode(nodeName, volumeName)

	// Assert
	if nodeErr != nil {
		t.Fatalf("AddNode failed. Expected: <no error> Actual: <%v>", nodeErr)
	}

	nodeExists, nodeExistsErr := vc.NodeExists(nodeName, volumeName)
	if nodeExistsErr != nil {
		t.Fatalf("NodeExists failed. Expected: <no error> Actual: <%v>", nodeExistsErr)
	}
	if !nodeExists {
		t.Fatalf("Added node %q does not exist, it should.", nodeName)
	}
}

func Test_AddNode_Positive_NodeExistsVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	vc.AddVolume(volumeName)
	nodeErr1 := vc.AddNode(nodeName, volumeName)
	if nodeErr1 != nil {
		t.Fatalf("First call to AddNode failed. Expected: <no error> Actual: <%v>", nodeErr1)
	}

	// Act
	nodeErr2 := vc.AddNode(nodeName, volumeName)

	// Assert
	if nodeErr2 != nil {
		t.Fatalf("Second call to AddNode failed. Expected: <no error> Actual: <%v>", nodeErr2)
	}

	nodeExists, nodeExistsErr := vc.NodeExists(nodeName, volumeName)
	if nodeExistsErr != nil {
		t.Fatalf("NodeExists failed. Expected: <no error> Actual: <%v>", nodeExistsErr)
	}
	if !nodeExists {
		t.Fatalf("Added node %q does not exist, it should.", nodeName)
	}
}

func Test_AddNode_Negative_NewNodeVolumeDoesntExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"

	// Act
	nodeErr := vc.AddNode(nodeName, volumeName)

	// Assert
	if nodeErr == nil {
		t.Fatalf("AddNode did not fail. Expected: <\"failed to add node...no volume with that name exists in the list of managed volumes\"> Actual: <no error>")
	}

	nodeExists, nodeExistsErr := vc.NodeExists(nodeName, volumeName)
	if nodeExistsErr == nil {
		t.Fatalf("NodeExists did not fail. Expected: <failed to check if node...no volume with that name exists in the list of managed volumes> Actual: <no error>")
	}
	if nodeExists {
		t.Fatalf("Added node %q exists, it should not.", nodeName)
	}
}

func Test_AddPod_Positive_NewPodNodeExistsVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	podName := "pod-name"
	vc.AddVolume(volumeName)
	nodeErr := vc.AddNode(nodeName, volumeName)
	if nodeErr != nil {
		t.Fatalf("AddNode failed. Expected: <no error> Actual: <%v>", nodeErr)
	}

	// Act
	podErr := vc.AddPod(podName, nodeName, volumeName)

	// Assert
	if podErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podErr)
	}

	podExists, podExistsErr := vc.PodExists(podName, nodeName, volumeName)
	if podExistsErr != nil {
		t.Fatalf("PodExists failed. Expected: <no error> Actual: <%v>", podExistsErr)
	}
	if !podExists {
		t.Fatalf("Added pod %q does not exist, it should.", podName)
	}
}

func Test_AddPod_Positive_PodExistsNodeExistsVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	podName := "pod-name"
	vc.AddVolume(volumeName)
	nodeErr := vc.AddNode(nodeName, volumeName)
	if nodeErr != nil {
		t.Fatalf("AddNode failed. Expected: <no error> Actual: <%v>", nodeErr)
	}
	podErr1 := vc.AddPod(podName, nodeName, volumeName)
	if podErr1 != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podErr1)
	}

	// Act
	podErr2 := vc.AddPod(podName, nodeName, volumeName)

	// Assert
	if podErr2 != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podErr2)
	}

	podExists, podExistsErr := vc.PodExists(podName, nodeName, volumeName)
	if podExistsErr != nil {
		t.Fatalf("PodExists failed. Expected: <no error> Actual: <%v>", podExistsErr)
	}
	if !podExists {
		t.Fatalf("Added pod %q does not exist, it should.", podName)
	}
}

func Test_AddPod_Negative_NewPodNodeDoesntExistsVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	podName := "pod-name"
	vc.AddVolume(volumeName)

	// Act
	podErr := vc.AddPod(podName, nodeName, volumeName)

	// Assert
	if podErr == nil {
		t.Fatalf("AddPod did not fail. Expected: <\"failed to add pod...no node with that name exists in the list of attached nodes for that volume\"> Actual: <no error>")
	}

	podExists, podExistsErr := vc.PodExists(podName, nodeName, volumeName)
	if podExistsErr == nil {
		t.Fatalf("PodExists did not fail. Expected: <\"failed to check if pod exists...no node with that name exists in the list of attached nodes for that volume\"> Actual: <no error>")
	}
	if podExists {
		t.Fatalf("Added pod %q exists, it should not.", podName)
	}
}

func Test_AddPod_Negative_NewPodNodeDoesntExistsVolumeDoesntExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	podName := "pod-name"

	// Act
	podErr := vc.AddPod(podName, nodeName, volumeName)

	// Assert
	if podErr == nil {
		t.Fatalf("AddPod did not fail. Expected: <\"failed to add pod...no volume with that name exists in the list of managed volumes\"> Actual: <no error>")
	}

	podExists, podExistsErr := vc.PodExists(podName, nodeName, volumeName)
	if podExistsErr == nil {
		t.Fatalf("PodExists did not fail. Expected: <\"failed to check if node...no volume with that name exists in the list of managed volumes\"> Actual: <no error>")
	}
	if podExists {
		t.Fatalf("Added pod %q exists, it should not.", podName)
	}
}

func Test_VolumeExists_Positive_NonExistantVolume(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	notAddedVolumeName := "volume-not-added-name"

	// Act
	notAddedVolumeExists := vc.VolumeExists(notAddedVolumeName)

	// Assert
	if notAddedVolumeExists {
		t.Fatalf("Not added volume %q exists, it should not.", notAddedVolumeName)
	}
}

func Test_NodeExists_Positive_NonExistantNodeVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	notAddedNodeName := "node-not-added-name"
	vc.AddVolume(volumeName)

	// Act
	notAddedNodeExists, notAddedNodeExistsErr := vc.NodeExists(notAddedNodeName, volumeName)

	// Assert
	if notAddedNodeExistsErr != nil {
		t.Fatalf("NodeExists failed. Expected: <no error> Actual: <%v>", notAddedNodeExistsErr)
	}
	if notAddedNodeExists {
		t.Fatalf("Not added node %q exists, it should not.", notAddedNodeName)
	}
}

func Test_NodeExists_Negative_NonExistantNodeVolumeDoesntExist(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	notAddedNodeName := "node-not-added-name"

	// Act
	notAddedNodeExists, notAddedNodeExistsErr := vc.NodeExists(notAddedNodeName, volumeName)

	// Assert
	if notAddedNodeExistsErr == nil {
		t.Fatalf("NodeExists did not fail. Expected: <failed to check if node...no volume with that name exists in the list of managed volumes> Actual: <no error>")
	}
	if notAddedNodeExists {
		t.Fatalf("Added node %q exists, it should not.", notAddedNodeName)
	}
}

func Test_PodExists_Positive_NonExistantPodNodeExistsVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	notAddedPodName := "pod-not-added-name"
	vc.AddVolume(volumeName)
	addNodeErr := vc.AddNode(nodeName, volumeName)
	if addNodeErr != nil {
		t.Fatalf("AddNode for node %q failed. Expected: <no error> Actual: <%v>", nodeName, addNodeErr)
	}

	// Act
	notAddedPodExists, notAddedPodExistsErr := vc.PodExists(notAddedPodName, nodeName, volumeName)

	// Assert
	if notAddedPodExistsErr != nil {
		t.Fatalf("PodExists failed. Expected: <no error> Actual: <%v>", notAddedPodExistsErr)
	}
	if notAddedPodExists {
		t.Fatalf("Not added pod %q exists, it should not.", notAddedPodName)
	}
}

func Test_PodExists_Negative_NonExistantPodNodeDoesntExistVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	notAddedPodName := "pod-not-added-name"
	vc.AddVolume(volumeName)

	// Act
	notAddedPodExists, notAddedPodExistsErr := vc.PodExists(notAddedPodName, nodeName, volumeName)

	// Assert
	if notAddedPodExistsErr == nil {
		t.Fatalf("PodExists did not fail. Expected: <\"failed to check if pod exists...no node with that name exists in the list of attached nodes for that volume\"> Actual: <no error>")
	}
	if notAddedPodExists {
		t.Fatalf("Added pod %q exists, it should not.", notAddedPodName)
	}
}

func Test_PodExists_Negative_NonExistantPodNodeDoesntExistVolumeDoesntExist(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	notAddedPodName := "pod-not-added-name"

	// Act
	notAddedPodExists, notAddedPodExistsErr := vc.PodExists(notAddedPodName, nodeName, volumeName)

	// Assert
	if notAddedPodExistsErr == nil {
		t.Fatalf("PodExists did not fail. Expected: <\"failed to check if node...no volume with that name exists in the list of managed volumes\"> Actual: <no error>")
	}
	if notAddedPodExists {
		t.Fatalf("Added pod %q exists, it should not.", notAddedPodName)
	}
}

func Test_DeleteVolume_Positive_VolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	vc.AddVolume(volumeName)

	// Act
	deleteVolumeErr := vc.DeleteVolume(volumeName)

	// Assert
	if deleteVolumeErr != nil {
		t.Fatalf("DeleteVolume failed. Expected: <no error> Actual: <%v>", deleteVolumeErr)
	}

	volumeExists := vc.VolumeExists(volumeName)
	if volumeExists {
		t.Fatalf("Deleted volume %q still exists, it should not.", volumeName)
	}
}

func Test_DeleteVolume_Negative_VolumeDoesntExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	notAddedVolumeName := "volume-not-added-name"

	// Act
	deleteVolumeErr := vc.DeleteVolume(notAddedVolumeName)

	// Assert
	if deleteVolumeErr == nil {
		t.Fatalf("DeleteVolume did not fail. Expected: <\"failed to delete volume...no volume with that name exists in the list of managed volumes\"> Actual: <no error>")
	}

	notAddedVolumeExists := vc.VolumeExists(notAddedVolumeName)
	if notAddedVolumeExists {
		t.Fatalf("Not added volume %q exists, it should not.", notAddedVolumeName)
	}
}

func Test_DeleteNode_Positive_NodeExistsVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	vc.AddVolume(volumeName)
	nodeErr := vc.AddNode(nodeName, volumeName)
	if nodeErr != nil {
		t.Fatalf("AddNode failed. Expected: <no error> Actual: <%v>", nodeErr)
	}

	// Act
	deleteNodeErr := vc.DeleteNode(nodeName, volumeName)

	// Assert
	if deleteNodeErr != nil {
		t.Fatalf("DeleteNode failed. Expected: <no error> Actual: <%v>", deleteNodeErr)
	}

	nodeExists, nodeExistsErr := vc.NodeExists(nodeName, volumeName)
	if nodeExistsErr != nil {
		t.Fatalf("NodeExists failed. Expected: <no error> Actual: <%v>", nodeExistsErr)
	}
	if nodeExists {
		t.Fatalf("Deleted node %q still exists, it should not.", nodeName)
	}
}

func Test_DeleteNode_Negative_NodeDoesntExistVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	notAddedNodeName := "node-not-added-name"
	vc.AddVolume(volumeName)

	// Act
	deleteNodeErr := vc.DeleteNode(notAddedNodeName, volumeName)

	// Assert
	if deleteNodeErr == nil {
		t.Fatalf("DeleteNode did not fail. Expected: <\"failed to delete node...no node with the that name exists in the list of attached nodes for that volume\"> Actual: <no error>")
	}

	notAddedNodeExists, notAddedNodeExistsErr := vc.NodeExists(notAddedNodeName, volumeName)
	if notAddedNodeExistsErr != nil {
		t.Fatalf("NodeExists failed. Expected: <no error> Actual: <%v>", notAddedNodeExistsErr)
	}
	if notAddedNodeExists {
		t.Fatalf("Not added node %q exists, it should not.", notAddedNodeName)
	}
}

func Test_DeleteNode_Negative_NodeDoesntExistVolumeDoesntExist(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	notAddedNodeName := "node-not-added-name"

	// Act
	deleteNodeErr := vc.DeleteNode(notAddedNodeName, volumeName)

	// Assert
	if deleteNodeErr == nil {
		t.Fatalf("DeleteNode did not fail. Expected: <\"failed to delete node...no volume with that name exists in the list of managed volumes\"> Actual: <no error>")
	}

	notAddedNodeExists, notAddedNodeExistsErr := vc.NodeExists(notAddedNodeName, volumeName)
	if notAddedNodeExistsErr == nil {
		t.Fatalf("NodeExists did not fail. Expected: <\failed to check if node...no volume with that name exists in the list of managed volumes\"> Actual: <no error>")
	}
	if notAddedNodeExists {
		t.Fatalf("Not added node %q exists, it should not.", notAddedNodeName)
	}
}

func Test_DeletePod_Positive_PodExistsNodeExistsVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	podName := "pod-name"
	vc.AddVolume(volumeName)
	nodeErr := vc.AddNode(nodeName, volumeName)
	if nodeErr != nil {
		t.Fatalf("AddNode failed. Expected: <no error> Actual: <%v>", nodeErr)
	}
	podErr := vc.AddPod(podName, nodeName, volumeName)
	if podErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podErr)
	}

	// Act
	deletePodErr := vc.DeletePod(podName, nodeName, volumeName)

	// Assert
	if deletePodErr != nil {
		t.Fatalf("DeletePod failed. Expected: <no error> Actual: <%v>", podName)
	}

	podExists, podExistsErr := vc.PodExists(podName, nodeName, volumeName)
	if podExistsErr != nil {
		t.Fatalf("PodExists failed. Expected: <no error> Actual: <%v>", podExistsErr)
	}
	if podExists {
		t.Fatalf("Deleted pod %q still exists, it should not.", podName)
	}
}

func Test_DeletePod_Positive_PodDoesntExistNodeExistsVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	podName := "pod-name"
	vc.AddVolume(volumeName)
	nodeErr := vc.AddNode(nodeName, volumeName)
	if nodeErr != nil {
		t.Fatalf("AddNode failed. Expected: <no error> Actual: <%v>", nodeErr)
	}

	// Act
	deletePodErr := vc.DeletePod(podName, nodeName, volumeName)

	// Assert
	if deletePodErr == nil {
		t.Fatalf("DeletePod did not fail. Expected: <\"failed to delete pod...no pod with that name exists in the list of scheduled pods under that node/volume\"> Actual: <no error>")
	}

	podExists, podExistsErr := vc.PodExists(podName, nodeName, volumeName)
	if podExistsErr != nil {
		t.Fatalf("PodExists failed. Expected: <no error> Actual: <%v>", podExistsErr)
	}
	if podExists {
		t.Fatalf("Deleted pod %q still exists, it should not.", podName)
	}
}

func Test_DeletePod_Positive_PodDoesntExistNodeDoesntExistVolumeExists(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	podName := "pod-name"
	vc.AddVolume(volumeName)

	// Act
	deletePodErr := vc.DeletePod(podName, nodeName, volumeName)

	// Assert
	if deletePodErr == nil {
		t.Fatalf("DeletePod did not fail. Expected: <\"failed to delete pod...no node with that name exists in the list of attached nodes for that volume\"> Actual: <no error>")
	}

	podExists, podExistsErr := vc.PodExists(podName, nodeName, volumeName)
	if podExistsErr == nil {
		t.Fatalf("PodExists did not fail. Expected: <\failed to check if pod...no node with that name exists in the list of attached nodes for that volume\"> Actual: <no error>")
	}
	if podExists {
		t.Fatalf("Deleted pod %q still exists, it should not.", podName)
	}
}

func Test_DeletePod_Positive_PodDoesntExistNodeDoesntExistVolumeDoesntExist(t *testing.T) {
	// Arrange
	vc := NewAttachDetachVolumeCache()
	volumeName := "volume-name"
	nodeName := "node-name"
	podName := "pod-name"

	// Act
	deletePodErr := vc.DeletePod(podName, nodeName, volumeName)

	// Assert
	if deletePodErr == nil {
		t.Fatalf("DeletePod did not fail. Expected: <\"failed to delete pod...no volume with that name exists in the list of managed volumes\"> Actual: <no error>")
	}

	podExists, podExistsErr := vc.PodExists(podName, nodeName, volumeName)
	if podExistsErr == nil {
		t.Fatalf("PodExists did not fail. Expected: <\failed to check if pod...no volume with that name exists in the list of managed volumes\"> Actual: <no error>")
	}
	if podExists {
		t.Fatalf("Deleted pod %q still exists, it should not.", podName)
	}
}

/*
	t.Fatalf("%q", notAddedNodeExistsErr)
*/
