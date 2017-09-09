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

package reconciler

import (
	"testing"
	"time"

	"k8s.io/api/core/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/statusupdater"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

const (
	reconcilerLoopPeriod      time.Duration = 0 * time.Millisecond
	syncLoopPeriod            time.Duration = 100 * time.Minute
	maxWaitForUnmountDuration time.Duration = 50 * time.Millisecond
	resyncPeriod              time.Duration = 5 * time.Minute
)

// Calls Run()
// Verifies there are no calls to attach or detach.
func Test_Run_Positive_DoNothing(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(fakeKubeClient, volumePluginMgr, fakeRecorder, false /* checkNodeCapabilitiesBeforeMount */))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nsu := statusupdater.NewNodeStatusUpdater(
		fakeKubeClient, informerFactory.Core().V1().Nodes().Lister(), asw)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, fakeRecorder)

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Assert
	waitForNewAttacherCallCount(t, 0 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, true /* expectZeroNewAttacherCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, true /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForAttachCallCount(t, 0 /* expectedAttachCallCount */, fakePlugin)
	waitForDetachCallCount(t, 0 /* expectedDetachCallCount */, fakePlugin)
}

// Populates desiredStateOfWorld cache with one node/volume/pod tuple.
// Calls Run()
// Verifies there is one attach call and no detach calls.
func Test_Run_Positive_OneDesiredVolumeAttach(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(fakeKubeClient, volumePluginMgr, fakeRecorder, false /* checkNodeCapabilitiesBeforeMount */))
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, fakeRecorder)
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

	_, podErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)
	if podErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podErr)
	}

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Assert
	waitForNewAttacherCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	waitForAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, true /* expectZeroNewDetacherCallCount */, fakePlugin)
}

// Populates desiredStateOfWorld cache with one node/volume/pod tuple.
// Calls Run()
// Verifies there is one attach call and no detach calls.
// Marks the node/volume as unmounted.
// Deletes the node/volume/pod tuple from desiredStateOfWorld cache.
// Verifies there is one detach call and no (new) attach calls.
func Test_Run_Positive_OneDesiredVolumeAttachThenDetachWithUnmountedVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(fakeKubeClient, volumePluginMgr, fakeRecorder, false /* checkNodeCapabilitiesBeforeMount */))
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, fakeRecorder)
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

	generatedVolumeName, podAddErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Assert
	waitForNewAttacherCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, true /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 0 /* expectedDetachCallCount */, fakePlugin)

	// Act
	dsw.DeletePod(types.UniquePodName(podName), generatedVolumeName, nodeName)
	volumeExists = dsw.VolumeExists(generatedVolumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Deleted pod %q from volume %q/node %q. Volume should also be deleted but it still exists.",
			podName,
			generatedVolumeName,
			nodeName)
	}
	asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, true /* mounted */)
	asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, false /* mounted */)

	// Assert
	waitForNewDetacherCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, false /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 1 /* expectedDetachCallCount */, fakePlugin)
}

// Populates desiredStateOfWorld cache with one node/volume/pod tuple.
// Calls Run()
// Verifies there is one attach call and no detach calls.
// Deletes the node/volume/pod tuple from desiredStateOfWorld cache without first marking the node/volume as unmounted.
// Verifies there is one detach call and no (new) attach calls.
func Test_Run_Positive_OneDesiredVolumeAttachThenDetachWithMountedVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(fakeKubeClient, volumePluginMgr, fakeRecorder, false /* checkNodeCapabilitiesBeforeMount */))
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, fakeRecorder)
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

	generatedVolumeName, podAddErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Assert
	waitForNewAttacherCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, true /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 0 /* expectedDetachCallCount */, fakePlugin)

	// Act
	dsw.DeletePod(types.UniquePodName(podName), generatedVolumeName, nodeName)
	volumeExists = dsw.VolumeExists(generatedVolumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Deleted pod %q from volume %q/node %q. Volume should also be deleted but it still exists.",
			podName,
			generatedVolumeName,
			nodeName)
	}

	// Assert -- Timer will triger detach
	waitForNewDetacherCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, false /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 1 /* expectedDetachCallCount */, fakePlugin)
}

// Populates desiredStateOfWorld cache with one node/volume/pod tuple.
// Has node update fail
// Calls Run()
// Verifies there is one attach call and no detach calls.
// Marks the node/volume as unmounted.
// Deletes the node/volume/pod tuple from desiredStateOfWorld cache.
// Verifies there are NO detach call and no (new) attach calls.
func Test_Run_Negative_OneDesiredVolumeAttachThenDetachWithUnmountedVolumeUpdateStatusFail(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(fakeKubeClient, volumePluginMgr, fakeRecorder, false /* checkNodeCapabilitiesBeforeMount */))
	nsu := statusupdater.NewFakeNodeStatusUpdater(true /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, fakeRecorder)
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

	generatedVolumeName, podAddErr := dsw.AddPod(types.UniquePodName(podName), controllervolumetesting.NewPod(podName, podName), volumeSpec, nodeName)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Assert
	waitForNewAttacherCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, true /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 0 /* expectedDetachCallCount */, fakePlugin)

	// Act
	dsw.DeletePod(types.UniquePodName(podName), generatedVolumeName, nodeName)
	volumeExists = dsw.VolumeExists(generatedVolumeName, nodeName)
	if volumeExists {
		t.Fatalf(
			"Deleted pod %q from volume %q/node %q. Volume should also be deleted but it still exists.",
			podName,
			generatedVolumeName,
			nodeName)
	}
	asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, true /* mounted */)
	asw.SetVolumeMountedByNode(generatedVolumeName, nodeName, false /* mounted */)

	// Assert
	verifyNewDetacherCallCount(t, true /* expectZeroNewDetacherCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, false /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 0 /* expectedDetachCallCount */, fakePlugin)
}

// Creates a volume with accessMode ReadWriteMany
// Populates desiredStateOfWorld cache with two ode/volume/pod tuples pointing to the created volume
// Calls Run()
// Verifies there are two attach calls and no detach calls.
// Deletes the first node/volume/pod tuple from desiredStateOfWorld cache without first marking the node/volume as unmounted.
// Verifies there is one detach call and no (new) attach calls.
// Deletes the second node/volume/pod tuple from desiredStateOfWorld cache without first marking the node/volume as unmounted.
// Verifies there are two detach calls and no (new) attach calls.
func Test_Run_OneVolumeAttachAndDetachMultipleNodesWithReadWriteMany(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(fakeKubeClient, volumePluginMgr, fakeRecorder, false /* checkNodeCapabilitiesBeforeMount */))
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, fakeRecorder)
	podName1 := "pod-uid1"
	podName2 := "pod-uid2"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	volumeSpec.PersistentVolume.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteMany}
	nodeName1 := k8stypes.NodeName("node-name1")
	nodeName2 := k8stypes.NodeName("node-name2")
	dsw.AddNode(nodeName1, false /*keepTerminatedPodVolumes*/)
	dsw.AddNode(nodeName2, false /*keepTerminatedPodVolumes*/)

	generatedVolumeName, podAddErr := dsw.AddPod(types.UniquePodName(podName1), controllervolumetesting.NewPod(podName1, podName1), volumeSpec, nodeName1)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}
	_, podAddErr = dsw.AddPod(types.UniquePodName(podName2), controllervolumetesting.NewPod(podName2, podName2), volumeSpec, nodeName2)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Assert
	waitForNewAttacherCallCount(t, 2 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForTotalAttachCallCount(t, 2 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, true /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 0 /* expectedDetachCallCount */, fakePlugin)
	waitForAttachedToNodesCount(t, 2 /* expectedNodeCount */, generatedVolumeName, asw)

	// Act
	dsw.DeletePod(types.UniquePodName(podName1), generatedVolumeName, nodeName1)
	volumeExists := dsw.VolumeExists(generatedVolumeName, nodeName1)
	if volumeExists {
		t.Fatalf(
			"Deleted pod %q from volume %q/node %q. Volume should also be deleted but it still exists.",
			podName1,
			generatedVolumeName,
			nodeName1)
	}

	// Assert -- Timer will triger detach
	waitForNewDetacherCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForTotalAttachCallCount(t, 2 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, false /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForTotalDetachCallCount(t, 1 /* expectedDetachCallCount */, fakePlugin)

	// Act
	dsw.DeletePod(types.UniquePodName(podName2), generatedVolumeName, nodeName2)
	volumeExists = dsw.VolumeExists(generatedVolumeName, nodeName2)
	if volumeExists {
		t.Fatalf(
			"Deleted pod %q from volume %q/node %q. Volume should also be deleted but it still exists.",
			podName2,
			generatedVolumeName,
			nodeName2)
	}

	// Assert -- Timer will triger detach
	waitForNewDetacherCallCount(t, 2 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForTotalAttachCallCount(t, 2 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, false /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForTotalDetachCallCount(t, 2 /* expectedDetachCallCount */, fakePlugin)
}

// Creates a volume with accessMode ReadWriteOnce
// Populates desiredStateOfWorld cache with two ode/volume/pod tuples pointing to the created volume
// Calls Run()
// Verifies there is one attach call and no detach calls.
// Deletes the node/volume/pod tuple from desiredStateOfWorld which succeeded in attaching
// Verifies there are two attach call and one detach call.
func Test_Run_OneVolumeAttachAndDetachMultipleNodesWithReadWriteOnce(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(fakeKubeClient, volumePluginMgr, fakeRecorder, false /* checkNodeCapabilitiesBeforeMount */))
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, fakeRecorder)
	podName1 := "pod-uid1"
	podName2 := "pod-uid2"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	volumeSpec.PersistentVolume.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
	nodeName1 := k8stypes.NodeName("node-name1")
	nodeName2 := k8stypes.NodeName("node-name2")
	dsw.AddNode(nodeName1, false /*keepTerminatedPodVolumes*/)
	dsw.AddNode(nodeName2, false /*keepTerminatedPodVolumes*/)

	// Add both pods at the same time to provoke a potential race condition in the reconciler
	generatedVolumeName, podAddErr := dsw.AddPod(types.UniquePodName(podName1), controllervolumetesting.NewPod(podName1, podName1), volumeSpec, nodeName1)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}
	_, podAddErr = dsw.AddPod(types.UniquePodName(podName2), controllervolumetesting.NewPod(podName2, podName2), volumeSpec, nodeName2)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Assert
	waitForNewAttacherCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForTotalAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, true /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 0 /* expectedDetachCallCount */, fakePlugin)
	waitForAttachedToNodesCount(t, 1 /* expectedNodeCount */, generatedVolumeName, asw)

	nodesForVolume := asw.GetNodesForVolume(generatedVolumeName)

	// Act
	podToDelete := ""
	if nodesForVolume[0] == nodeName1 {
		podToDelete = podName1
	} else if nodesForVolume[0] == nodeName2 {
		podToDelete = podName2
	} else {
		t.Fatal("Volume attached to unexpected node")
	}

	dsw.DeletePod(types.UniquePodName(podToDelete), generatedVolumeName, nodesForVolume[0])
	volumeExists := dsw.VolumeExists(generatedVolumeName, nodesForVolume[0])
	if volumeExists {
		t.Fatalf(
			"Deleted pod %q from volume %q/node %q. Volume should also be deleted but it still exists.",
			podToDelete,
			generatedVolumeName,
			nodesForVolume[0])
	}

	// Assert
	waitForNewDetacherCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, false /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForTotalDetachCallCount(t, 1 /* expectedDetachCallCount */, fakePlugin)
	waitForNewAttacherCallCount(t, 2 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForTotalAttachCallCount(t, 2 /* expectedAttachCallCount */, fakePlugin)
}

func waitForNewAttacherCallCount(
	t *testing.T,
	expectedCallCount int,
	fakePlugin *volumetesting.FakeVolumePlugin) {
	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			actualCallCount := fakePlugin.GetNewAttacherCallCount()
			if actualCallCount >= expectedCallCount {
				return true, nil
			}
			t.Logf(
				"Warning: Wrong NewAttacherCallCount. Expected: <%v> Actual: <%v>. Will retry.",
				expectedCallCount,
				actualCallCount)
			return false, nil
		},
	)

	if err != nil {
		t.Fatalf(
			"Timed out waiting for NewAttacherCallCount. Expected: <%v> Actual: <%v>",
			expectedCallCount,
			fakePlugin.GetNewAttacherCallCount())
	}
}

func waitForNewDetacherCallCount(
	t *testing.T,
	expectedCallCount int,
	fakePlugin *volumetesting.FakeVolumePlugin) {
	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			actualCallCount := fakePlugin.GetNewDetacherCallCount()
			if actualCallCount >= expectedCallCount {
				return true, nil
			}
			t.Logf(
				"Warning: Wrong NewDetacherCallCount. Expected: <%v> Actual: <%v>. Will retry.",
				expectedCallCount,
				actualCallCount)
			return false, nil
		},
	)

	if err != nil {
		t.Fatalf(
			"Timed out waiting for NewDetacherCallCount. Expected: <%v> Actual: <%v>",
			expectedCallCount,
			fakePlugin.GetNewDetacherCallCount())
	}
}

func waitForAttachCallCount(
	t *testing.T,
	expectedAttachCallCount int,
	fakePlugin *volumetesting.FakeVolumePlugin) {
	if len(fakePlugin.GetAttachers()) == 0 && expectedAttachCallCount == 0 {
		return
	}

	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			for i, attacher := range fakePlugin.GetAttachers() {
				actualCallCount := attacher.GetAttachCallCount()
				if actualCallCount == expectedAttachCallCount {
					return true, nil
				}
				t.Logf(
					"Warning: Wrong attacher[%v].GetAttachCallCount(). Expected: <%v> Actual: <%v>. Will try next attacher.",
					i,
					expectedAttachCallCount,
					actualCallCount)
			}

			t.Logf(
				"Warning: No attachers have expected AttachCallCount. Expected: <%v>. Will retry.",
				expectedAttachCallCount)
			return false, nil
		},
	)

	if err != nil {
		t.Fatalf(
			"No attachers have expected AttachCallCount. Expected: <%v>",
			expectedAttachCallCount)
	}
}

func waitForTotalAttachCallCount(
	t *testing.T,
	expectedAttachCallCount int,
	fakePlugin *volumetesting.FakeVolumePlugin) {
	if len(fakePlugin.GetAttachers()) == 0 && expectedAttachCallCount == 0 {
		return
	}

	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			totalCount := 0
			for _, attacher := range fakePlugin.GetAttachers() {
				totalCount += attacher.GetAttachCallCount()
			}
			if totalCount == expectedAttachCallCount {
				return true, nil
			}
			t.Logf(
				"Warning: Wrong total GetAttachCallCount(). Expected: <%v> Actual: <%v>. Will retry.",
				expectedAttachCallCount,
				totalCount)

			return false, nil
		},
	)

	if err != nil {
		t.Fatalf(
			"Total AttachCallCount does not match expected value. Expected: <%v>",
			expectedAttachCallCount)
	}
}

func waitForDetachCallCount(
	t *testing.T,
	expectedDetachCallCount int,
	fakePlugin *volumetesting.FakeVolumePlugin) {
	if len(fakePlugin.GetDetachers()) == 0 && expectedDetachCallCount == 0 {
		return
	}

	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			for i, detacher := range fakePlugin.GetDetachers() {
				actualCallCount := detacher.GetDetachCallCount()
				if actualCallCount == expectedDetachCallCount {
					return true, nil
				}
				t.Logf(
					"Wrong detacher[%v].GetDetachCallCount(). Expected: <%v> Actual: <%v>. Will try next detacher.",
					i,
					expectedDetachCallCount,
					actualCallCount)
			}

			t.Logf(
				"Warning: No detachers have expected DetachCallCount. Expected: <%v>. Will retry.",
				expectedDetachCallCount)
			return false, nil
		},
	)

	if err != nil {
		t.Fatalf(
			"No detachers have expected DetachCallCount. Expected: <%v>",
			expectedDetachCallCount)
	}
}

func waitForTotalDetachCallCount(
	t *testing.T,
	expectedDetachCallCount int,
	fakePlugin *volumetesting.FakeVolumePlugin) {
	if len(fakePlugin.GetDetachers()) == 0 && expectedDetachCallCount == 0 {
		return
	}

	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			totalCount := 0
			for _, detacher := range fakePlugin.GetDetachers() {
				totalCount += detacher.GetDetachCallCount()
			}
			if totalCount == expectedDetachCallCount {
				return true, nil
			}
			t.Logf(
				"Warning: Wrong total GetDetachCallCount(). Expected: <%v> Actual: <%v>. Will retry.",
				expectedDetachCallCount,
				totalCount)

			return false, nil
		},
	)

	if err != nil {
		t.Fatalf(
			"Total DetachCallCount does not match expected value. Expected: <%v>",
			expectedDetachCallCount)
	}
}

func waitForAttachedToNodesCount(
	t *testing.T,
	expectedNodeCount int,
	volumeName v1.UniqueVolumeName,
	asw cache.ActualStateOfWorld) {

	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			count := len(asw.GetNodesForVolume(volumeName))
			if count == expectedNodeCount {
				return true, nil
			}
			t.Logf(
				"Warning: Wrong number of nodes having <%v> attached. Expected: <%v> Actual: <%v>. Will retry.",
				volumeName,
				expectedNodeCount,
				count)

			return false, nil
		},
	)

	if err != nil {
		count := len(asw.GetNodesForVolume(volumeName))
		t.Fatalf(
			"Wrong number of nodes having <%v> attached. Expected: <%v> Actual: <%v>",
			volumeName,
			expectedNodeCount,
			count)
	}
}

func verifyNewAttacherCallCount(
	t *testing.T,
	expectZeroNewAttacherCallCount bool,
	fakePlugin *volumetesting.FakeVolumePlugin) {

	if expectZeroNewAttacherCallCount &&
		fakePlugin.GetNewAttacherCallCount() != 0 {
		t.Fatalf(
			"Wrong NewAttacherCallCount. Expected: <0> Actual: <%v>",
			fakePlugin.GetNewAttacherCallCount())
	}
}

func verifyNewDetacherCallCount(
	t *testing.T,
	expectZeroNewDetacherCallCount bool,
	fakePlugin *volumetesting.FakeVolumePlugin) {

	if expectZeroNewDetacherCallCount &&
		fakePlugin.GetNewDetacherCallCount() != 0 {
		t.Fatalf("Wrong NewDetacherCallCount. Expected: <0> Actual: <%v>",
			fakePlugin.GetNewDetacherCallCount())
	}
}

func retryWithExponentialBackOff(initialDuration time.Duration, fn wait.ConditionFunc) error {
	backoff := wait.Backoff{
		Duration: initialDuration,
		Factor:   3,
		Jitter:   0,
		Steps:    6,
	}
	return wait.ExponentialBackoff(backoff, fn)
}
