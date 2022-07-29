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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/statusupdater"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
	"k8s.io/kubernetes/pkg/features"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/types"
	utilstrings "k8s.io/utils/strings"
)

const (
	reconcilerLoopPeriod          time.Duration = 10 * time.Millisecond
	syncLoopPeriod                time.Duration = 100 * time.Minute
	maxWaitForUnmountDuration     time.Duration = 50 * time.Millisecond
	maxLongWaitForUnmountDuration time.Duration = 4200 * time.Second
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
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nsu := statusupdater.NewNodeStatusUpdater(
		fakeKubeClient, informerFactory.Core().V1().Nodes().Lister(), asw)
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, nodeLister, fakeRecorder)

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
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, nodeLister, fakeRecorder)
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
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, nodeLister, fakeRecorder)
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
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, nodeLister, fakeRecorder)
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

	// Assert -- Timer will trigger detach
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
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	nsu := statusupdater.NewFakeNodeStatusUpdater(true /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, nodeLister, fakeRecorder)
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
// Populates desiredStateOfWorld cache with two node/volume/pod tuples pointing to the created volume
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
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, nodeLister, fakeRecorder)
	podName1 := "pod-uid1"
	podName2 := "pod-uid2"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	volumeSpec.PersistentVolume.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteMany}
	nodeName1 := k8stypes.NodeName("node-name1")
	nodeName2 := k8stypes.NodeName(volumetesting.MultiAttachNode)
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

	// Assert -- Timer will trigger detach
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

	// Assert -- Timer will trigger detach
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
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, nodeLister, fakeRecorder)
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

	nodesForVolume := asw.GetNodesForAttachedVolume(generatedVolumeName)

	// check if multiattach is marked
	// at least one volume+node should be marked with multiattach error
	nodeAttachedTo := nodesForVolume[0]
	waitForMultiAttachErrorOnNode(t, nodeAttachedTo, dsw)

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

// Creates a volume with accessMode ReadWriteOnce
// First create a pod which will try to attach the volume to the a node named "uncertain-node". The attach call for this node will
// fail for timeout, but the volume will be actually attached to the node after the call.
// Secondly, delete this pod.
// Lastly, create a pod scheduled to a normal node which will trigger attach volume to the node. The attach should return successfully.
func Test_Run_OneVolumeAttachAndDetachUncertainNodesWithReadWriteOnce(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, nodeLister, fakeRecorder)
	podName1 := "pod-uid1"
	podName2 := "pod-uid2"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	volumeSpec.PersistentVolume.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
	nodeName1 := k8stypes.NodeName(volumetesting.UncertainAttachNode)
	nodeName2 := k8stypes.NodeName("node-name2")
	dsw.AddNode(nodeName1, false /*keepTerminatedPodVolumes*/)
	dsw.AddNode(nodeName2, false /*keepTerminatedPodVolumes*/)

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Add the pod in which the volume is attached to the uncertain node
	generatedVolumeName, podAddErr := dsw.AddPod(types.UniquePodName(podName1), controllervolumetesting.NewPod(podName1, podName1), volumeSpec, nodeName1)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}

	time.Sleep(1 * time.Second)
	// Volume is added to asw. Because attach operation fails, volume should not be reported as attached to the node.
	waitForVolumeAddedToNode(t, generatedVolumeName, nodeName1, asw)
	verifyVolumeAttachedToNode(t, generatedVolumeName, nodeName1, cache.AttachStateAttached, asw)
	verifyVolumeReportedAsAttachedToNode(t, generatedVolumeName, nodeName1, true, asw)

	// When volume is added to the node, it is set to mounted by default. Then the status will be updated by checking node status VolumeInUse.
	// Without this, the delete operation will be delayed due to mounted status
	asw.SetVolumeMountedByNode(generatedVolumeName, nodeName1, false /* mounted */)

	dsw.DeletePod(types.UniquePodName(podName1), generatedVolumeName, nodeName1)

	waitForVolumeRemovedFromNode(t, generatedVolumeName, nodeName1, asw)

	// Add a second pod which tries to attach the volume to a different node.
	generatedVolumeName, podAddErr = dsw.AddPod(types.UniquePodName(podName2), controllervolumetesting.NewPod(podName2, podName2), volumeSpec, nodeName2)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}
	waitForVolumeAttachedToNode(t, generatedVolumeName, nodeName2, asw)
	verifyVolumeAttachedToNode(t, generatedVolumeName, nodeName2, cache.AttachStateAttached, asw)

}

func Test_Run_OneVolumeDetachFailNodeWithReadWriteOnce(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, nodeLister, fakeRecorder)
	podName1 := "pod-uid1"
	podName2 := "pod-uid2"
	podName3 := "pod-uid3"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	volumeSpec.PersistentVolume.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
	nodeName1 := k8stypes.NodeName(volumetesting.FailDetachNode)
	nodeName2 := k8stypes.NodeName("node-name2")
	dsw.AddNode(nodeName1, false /*keepTerminatedPodVolumes*/)
	dsw.AddNode(nodeName2, false /*keepTerminatedPodVolumes*/)

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Add the pod in which the volume is attached to the FailDetachNode
	generatedVolumeName, podAddErr := dsw.AddPod(types.UniquePodName(podName1), controllervolumetesting.NewPod(podName1, podName1), volumeSpec, nodeName1)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}

	time.Sleep(1000 * time.Millisecond)
	// Volume is added to asw, volume should be reported as attached to the node.
	waitForVolumeAddedToNode(t, generatedVolumeName, nodeName1, asw)
	verifyVolumeAttachedToNode(t, generatedVolumeName, nodeName1, cache.AttachStateAttached, asw)
	verifyVolumeReportedAsAttachedToNode(t, generatedVolumeName, nodeName1, true, asw)

	// Delete the pod, but detach will fail
	dsw.DeletePod(types.UniquePodName(podName1), generatedVolumeName, nodeName1)

	// The first detach will be triggered after at leaset 50ms (maxWaitForUnmountDuration in test).
	// Right before detach operation is performed, the volume will be first removed from being reported
	// as attached on node status (RemoveVolumeFromReportAsAttached). After detach operation which is expected to fail,
	// controller then added the volume back as attached.
	// Here it sleeps 100ms so that detach should be triggered already at this point.
	// verifyVolumeReportedAsAttachedToNode will check volume is in the list of volume attached that needs to be updated
	// in node status. By calling this function (GetVolumesToReportAttached), node status should be updated, and the volume
	// will not need to be updated until new changes are applied (detach is triggered again)
	time.Sleep(100 * time.Millisecond)
	verifyVolumeAttachedToNode(t, generatedVolumeName, nodeName1, cache.AttachStateAttached, asw)
	verifyVolumeReportedAsAttachedToNode(t, generatedVolumeName, nodeName1, true, asw)

	// After the first detach fails, reconciler will wait for a period of time before retrying to detach.
	// The wait time is increasing exponentially from initial value of 0.5s (0.5, 1, 2, 4, ...).
	// The test here waits for 100 Millisecond to make sure it is in exponential backoff period after
	// the first detach operation. At this point, volumes status should not be updated
	time.Sleep(100 * time.Millisecond)
	verifyVolumeAttachedToNode(t, generatedVolumeName, nodeName1, cache.AttachStateAttached, asw)
	verifyVolumeNoStatusUpdateNeeded(t, generatedVolumeName, nodeName1, asw)

	// Wait for 600ms to make sure second detach operation triggered. Again, The volume will be
	// removed from being reported as attached on node status and then added back as attached.
	// The volume will be in the list of attached volumes that need to be updated to node status.
	time.Sleep(600 * time.Millisecond)
	verifyVolumeAttachedToNode(t, generatedVolumeName, nodeName1, cache.AttachStateAttached, asw)
	verifyVolumeReportedAsAttachedToNode(t, generatedVolumeName, nodeName1, true, asw)

	// Add a second pod which tries to attach the volume to the same node.
	// After adding pod to the same node, detach will not be triggered any more.
	generatedVolumeName, podAddErr = dsw.AddPod(types.UniquePodName(podName2), controllervolumetesting.NewPod(podName2, podName2), volumeSpec, nodeName1)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}
	// Sleep 1s to verify no detach are triggered after second pod is added in the future.
	time.Sleep(1000 * time.Millisecond)
	verifyVolumeAttachedToNode(t, generatedVolumeName, nodeName1, cache.AttachStateAttached, asw)
	verifyVolumeNoStatusUpdateNeeded(t, generatedVolumeName, nodeName1, asw)

	// Add a third pod which tries to attach the volume to a different node.
	// At this point, volume is still attached to first node. There are no status update for both nodes.
	generatedVolumeName, podAddErr = dsw.AddPod(types.UniquePodName(podName3), controllervolumetesting.NewPod(podName3, podName3), volumeSpec, nodeName2)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}
	verifyVolumeAttachedToNode(t, generatedVolumeName, nodeName1, cache.AttachStateAttached, asw)
	verifyVolumeNoStatusUpdateNeeded(t, generatedVolumeName, nodeName1, asw)
	verifyVolumeNoStatusUpdateNeeded(t, generatedVolumeName, nodeName2, asw)
}

// Creates a volume with accessMode ReadWriteOnce
// First create a pod which will try to attach the volume to the a node named "timeout-node". The attach call for this node will
// fail for timeout, but the volume will be actually attached to the node after the call.
// Secondly, delete the this pod.
// Lastly, create a pod scheduled to a normal node which will trigger attach volume to the node. The attach should return successfully.
func Test_Run_OneVolumeAttachAndDetachTimeoutNodesWithReadWriteOnce(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, nodeLister, fakeRecorder)
	podName1 := "pod-uid1"
	podName2 := "pod-uid2"
	volumeName := v1.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	volumeSpec.PersistentVolume.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
	nodeName1 := k8stypes.NodeName(volumetesting.TimeoutAttachNode)
	nodeName2 := k8stypes.NodeName("node-name2")
	dsw.AddNode(nodeName1, false /*keepTerminatedPodVolumes*/)
	dsw.AddNode(nodeName2, false /*keepTerminatedPodVolumes*/)

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Add the pod in which the volume is attached to the timeout node
	generatedVolumeName, podAddErr := dsw.AddPod(types.UniquePodName(podName1), controllervolumetesting.NewPod(podName1, podName1), volumeSpec, nodeName1)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}

	// Volume is added to asw. Because attach operation fails, volume should not be reported as attached to the node.
	waitForVolumeAddedToNode(t, generatedVolumeName, nodeName1, asw)
	verifyVolumeAttachedToNode(t, generatedVolumeName, nodeName1, cache.AttachStateUncertain, asw)
	verifyVolumeReportedAsAttachedToNode(t, generatedVolumeName, nodeName1, false, asw)

	// When volume is added to the node, it is set to mounted by default. Then the status will be updated by checking node status VolumeInUse.
	// Without this, the delete operation will be delayed due to mounted status
	asw.SetVolumeMountedByNode(generatedVolumeName, nodeName1, false /* mounted */)

	dsw.DeletePod(types.UniquePodName(podName1), generatedVolumeName, nodeName1)

	waitForVolumeRemovedFromNode(t, generatedVolumeName, nodeName1, asw)

	// Add a second pod which tries to attach the volume to a different node.
	generatedVolumeName, podAddErr = dsw.AddPod(types.UniquePodName(podName2), controllervolumetesting.NewPod(podName2, podName2), volumeSpec, nodeName2)
	if podAddErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podAddErr)
	}
	waitForVolumeAttachedToNode(t, generatedVolumeName, nodeName2, asw)
	verifyVolumeAttachedToNode(t, generatedVolumeName, nodeName2, cache.AttachStateAttached, asw)

}

// Populates desiredStateOfWorld cache with one node/volume/pod tuple.
// The node has node.kubernetes.io/out-of-service taint present.
//
// The maxWaitForUnmountDuration is longer (in this case it is 4200 * time.Second so that detach does not happen
// immediately due to timeout.
//
// Calls Run()
// Verifies there is one attach call and no detach calls.
// Deletes the pod from desiredStateOfWorld cache without first marking the node/volume as unmounted.
// Verifies there is one detach call and no (new) attach calls.
func Test_Run_OneVolumeDetachOnOutOfServiceTaintedNode(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeOutOfServiceVolumeDetach, true)()
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxLongWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad,
		nsu, nodeLister, fakeRecorder)
	podName1 := "pod-uid1"
	volumeName1 := v1.UniqueVolumeName("volume-name1")
	volumeSpec1 := controllervolumetesting.GetTestVolumeSpec(string(volumeName1), volumeName1)
	nodeName1 := k8stypes.NodeName("worker-0")
	node1 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: string(nodeName1)},
		Spec: v1.NodeSpec{
			Taints: []v1.Taint{{Key: v1.TaintNodeOutOfService, Effect: v1.TaintEffectNoExecute}},
		},
	}
	informerFactory.Core().V1().Nodes().Informer().GetStore().Add(node1)
	dsw.AddNode(nodeName1, false /*keepTerminatedPodVolumes*/)
	volumeExists := dsw.VolumeExists(volumeName1, nodeName1)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName1,
			nodeName1)
	}

	generatedVolumeName, podErr := dsw.AddPod(types.UniquePodName(podName1), controllervolumetesting.NewPod(podName1,
		podName1), volumeSpec1, nodeName1)
	if podErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podErr)
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

	// Delete the pod and the volume will be detached only after the maxLongWaitForUnmountDuration expires as volume is
	//not unmounted. Here maxLongWaitForUnmountDuration is used to mimic that node is out of service.
	// But in this case the node has the node.kubernetes.io/out-of-service taint and hence it will not wait for
	// maxLongWaitForUnmountDuration and will progress to detach immediately.
	dsw.DeletePod(types.UniquePodName(podName1), generatedVolumeName, nodeName1)
	// Assert -- Detach will be triggered if node has out of service taint
	waitForNewDetacherCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, false /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 1 /* expectedDetachCallCount */, fakePlugin)
}

// Populates desiredStateOfWorld cache with one node/volume/pod tuple.
// The node does not have the node.kubernetes.io/out-of-service taint present.
//
// The maxWaitForUnmountDuration is longer (in this case it is 4200 * time.Second so that detach does not happen
// immediately due to timeout.
//
// Calls Run()
// Verifies there is one attach call and no detach calls.
// Deletes the pod from desiredStateOfWorld cache without first marking the node/volume as unmounted.
// Verifies there is no detach call and no (new) attach calls.
func Test_Run_OneVolumeDetachOnNoOutOfServiceTaintedNode(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeOutOfServiceVolumeDetach, true)()
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxLongWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad,
		nsu, nodeLister, fakeRecorder)
	podName1 := "pod-uid1"
	volumeName1 := v1.UniqueVolumeName("volume-name1")
	volumeSpec1 := controllervolumetesting.GetTestVolumeSpec(string(volumeName1), volumeName1)
	nodeName1 := k8stypes.NodeName("worker-0")
	node1 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: string(nodeName1)},
	}
	informerFactory.Core().V1().Nodes().Informer().GetStore().Add(node1)
	dsw.AddNode(nodeName1, false /*keepTerminatedPodVolumes*/)
	volumeExists := dsw.VolumeExists(volumeName1, nodeName1)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName1,
			nodeName1)
	}

	generatedVolumeName, podErr := dsw.AddPod(types.UniquePodName(podName1), controllervolumetesting.NewPod(podName1,
		podName1), volumeSpec1, nodeName1)
	if podErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podErr)
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

	// Delete the pod and the volume will be detached only after the maxLongWaitForUnmountDuration expires as volume is
	// not unmounted. Here maxLongWaitForUnmountDuration is used to mimic that node is out of service.
	// But in this case the node does not have the node.kubernetes.io/out-of-service taint and hence it will wait for
	// maxLongWaitForUnmountDuration and will not be detached immediately.
	dsw.DeletePod(types.UniquePodName(podName1), generatedVolumeName, nodeName1)
	// Assert -- Detach will be triggered only after maxLongWaitForUnmountDuration expires
	waitForNewDetacherCallCount(t, 0 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, true /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 0 /* expectedDetachCallCount */, fakePlugin)
}

// Populates desiredStateOfWorld cache with one node/volume/pod tuple.
// The node starts as healthy.
//
// Calls Run()
// Verifies there is one attach call and no detach calls.
// Deletes the pod from desiredStateOfWorld cache without first marking the node/volume as unmounted.
// Verifies that the volume is NOT detached after maxWaitForUnmountDuration.
// Marks the node as unhealthy.
// Verifies that the volume is detached after maxWaitForUnmountDuration.
func Test_Run_OneVolumeDetachOnUnhealthyNode(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad,
		nsu, nodeLister, fakeRecorder)
	podName1 := "pod-uid1"
	volumeName1 := v1.UniqueVolumeName("volume-name1")
	volumeSpec1 := controllervolumetesting.GetTestVolumeSpec(string(volumeName1), volumeName1)
	nodeName1 := k8stypes.NodeName("worker-0")
	node1 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: string(nodeName1)},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
		},
	}
	informerFactory.Core().V1().Nodes().Informer().GetStore().Add(node1)
	dsw.AddNode(nodeName1, false /*keepTerminatedPodVolumes*/)
	volumeExists := dsw.VolumeExists(volumeName1, nodeName1)
	if volumeExists {
		t.Fatalf(
			"Volume %q/node %q should not exist, but it does.",
			volumeName1,
			nodeName1)
	}

	generatedVolumeName, podErr := dsw.AddPod(types.UniquePodName(podName1), controllervolumetesting.NewPod(podName1,
		podName1), volumeSpec1, nodeName1)
	if podErr != nil {
		t.Fatalf("AddPod failed. Expected: <no error> Actual: <%v>", podErr)
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
	// Delete the pod and the volume will be detached even after the maxWaitForUnmountDuration expires as volume is
	// not unmounted and the node is healthy.
	dsw.DeletePod(types.UniquePodName(podName1), generatedVolumeName, nodeName1)
	time.Sleep(maxWaitForUnmountDuration * 5)
	// Assert
	waitForNewDetacherCallCount(t, 0 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, true /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 0 /* expectedDetachCallCount */, fakePlugin)

	// Act
	// Mark the node unhealthy
	node2 := node1.DeepCopy()
	node2.Status.Conditions[0].Status = v1.ConditionFalse
	informerFactory.Core().V1().Nodes().Informer().GetStore().Update(node2)
	// Assert -- Detach was triggered after maxWaitForUnmountDuration
	waitForNewDetacherCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	verifyNewAttacherCallCount(t, false /* expectZeroNewAttacherCallCount */, fakePlugin)
	waitForAttachCallCount(t, 1 /* expectedAttachCallCount */, fakePlugin)
	verifyNewDetacherCallCount(t, false /* expectZeroNewDetacherCallCount */, fakePlugin)
	waitForDetachCallCount(t, 1 /* expectedDetachCallCount */, fakePlugin)
}

func Test_ReportMultiAttachError(t *testing.T) {
	type nodeWithPods struct {
		name     k8stypes.NodeName
		podNames []string
	}
	tests := []struct {
		name           string
		nodes          []nodeWithPods
		expectedEvents []string
	}{
		{
			"no pods use the volume",
			[]nodeWithPods{
				{"node1", []string{"ns1/pod1"}},
			},
			[]string{"Warning FailedAttachVolume Multi-Attach error for volume \"volume-name\" Volume is already exclusively attached to one node and can't be attached to another"},
		},
		{
			"pods in the same namespace use the volume",
			[]nodeWithPods{
				{"node1", []string{"ns1/pod1"}},
				{"node2", []string{"ns1/pod2"}},
			},
			[]string{"Warning FailedAttachVolume Multi-Attach error for volume \"volume-name\" Volume is already used by pod(s) pod2"},
		},
		{
			"pods in another namespace use the volume",
			[]nodeWithPods{
				{"node1", []string{"ns1/pod1"}},
				{"node2", []string{"ns2/pod2"}},
			},
			[]string{"Warning FailedAttachVolume Multi-Attach error for volume \"volume-name\" Volume is already used by 1 pod(s) in different namespaces"},
		},
		{
			"pods both in the same and another namespace use the volume",
			[]nodeWithPods{
				{"node1", []string{"ns1/pod1"}},
				{"node2", []string{"ns2/pod2"}},
				{"node3", []string{"ns1/pod3"}},
			},
			[]string{"Warning FailedAttachVolume Multi-Attach error for volume \"volume-name\" Volume is already used by pod(s) pod3 and 1 pod(s) in different namespaces"},
		},
	}

	for _, test := range tests {
		// Arrange
		t.Logf("Test %q starting", test.name)
		volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
		dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
		asw := cache.NewActualStateOfWorld(volumePluginMgr)
		fakeKubeClient := controllervolumetesting.CreateTestClient()
		fakeRecorder := record.NewFakeRecorder(100)
		fakeHandler := volumetesting.NewBlockVolumePathHandler()
		ad := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
			fakeKubeClient,
			volumePluginMgr,
			fakeRecorder,
			fakeHandler))
		informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
		nodeLister := informerFactory.Core().V1().Nodes().Lister()
		nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
		rc := NewReconciler(
			reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, false, dsw, asw, ad, nsu, nodeLister, fakeRecorder)

		nodes := []k8stypes.NodeName{}
		for _, n := range test.nodes {
			dsw.AddNode(n.name, false /*keepTerminatedPodVolumes*/)
			nodes = append(nodes, n.name)
			for _, podName := range n.podNames {
				volumeName := v1.UniqueVolumeName("volume-name")
				volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
				volumeSpec.PersistentVolume.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				uid := string(n.name) + "-" + podName // unique UID
				namespace, name := utilstrings.SplitQualifiedName(podName)
				pod := controllervolumetesting.NewPod(uid, name)
				pod.Namespace = namespace
				_, err := dsw.AddPod(types.UniquePodName(uid), pod, volumeSpec, n.name)
				if err != nil {
					t.Fatalf("Error adding pod %s to DSW: %s", podName, err)
				}
			}
		}
		// Act
		volumes := dsw.GetVolumesToAttach()
		for _, vol := range volumes {
			if vol.NodeName == "node1" {
				rc.(*reconciler).reportMultiAttachError(vol, nodes)
			}
		}

		// Assert
		close(fakeRecorder.Events)
		index := 0
		for event := range fakeRecorder.Events {
			if len(test.expectedEvents) < index {
				t.Errorf("Test %q: unexpected event received: %s", test.name, event)
			} else {
				expectedEvent := test.expectedEvents[index]
				if expectedEvent != event {
					t.Errorf("Test %q: event %d: expected %q, got %q", test.name, index, expectedEvent, event)
				}
			}
			index++
		}
		for i := index; i < len(test.expectedEvents); i++ {
			t.Errorf("Test %q: event %d: expected %q, got none", test.name, i, test.expectedEvents[i])
		}
	}
}

func waitForMultiAttachErrorOnNode(
	t *testing.T,
	attachedNode k8stypes.NodeName,
	dsow cache.DesiredStateOfWorld) {
	multAttachCheckFunc := func() (bool, error) {
		for _, volumeToAttach := range dsow.GetVolumesToAttach() {
			if volumeToAttach.NodeName != attachedNode {
				if volumeToAttach.MultiAttachErrorReported {
					return true, nil
				}
			}
		}
		t.Logf("Warning: MultiAttach error not yet set on Node. Will retry.")
		return false, nil
	}

	err := retryWithExponentialBackOff(100*time.Millisecond, multAttachCheckFunc)
	if err != nil {
		t.Fatalf("Timed out waiting for MultiAttach Error to be set on non-attached node")
	}
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
			count := len(asw.GetNodesForAttachedVolume(volumeName))
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
		count := len(asw.GetNodesForAttachedVolume(volumeName))
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

func waitForVolumeAttachedToNode(
	t *testing.T,
	volumeName v1.UniqueVolumeName,
	nodeName k8stypes.NodeName,
	asw cache.ActualStateOfWorld) {

	err := retryWithExponentialBackOff(
		time.Duration(500*time.Millisecond),
		func() (bool, error) {
			attachState := asw.GetAttachState(volumeName, nodeName)
			if attachState == cache.AttachStateAttached {
				return true, nil
			}
			t.Logf(
				"Warning: Volume <%v> is not attached to node  <%v> yet. Will retry.",
				volumeName,
				nodeName)

			return false, nil
		},
	)

	attachState := asw.GetAttachState(volumeName, nodeName)
	if err != nil && attachState != cache.AttachStateAttached {
		t.Fatalf(
			"Volume <%v> is not attached to node  <%v>.",
			volumeName,
			nodeName)
	}
}

func waitForVolumeAddedToNode(
	t *testing.T,
	volumeName v1.UniqueVolumeName,
	nodeName k8stypes.NodeName,
	asw cache.ActualStateOfWorld) {

	err := retryWithExponentialBackOff(
		time.Duration(500*time.Millisecond),
		func() (bool, error) {
			volumes := asw.GetAttachedVolumes()
			for _, volume := range volumes {
				if volume.VolumeName == volumeName && volume.NodeName == nodeName {
					return true, nil
				}
			}
			t.Logf(
				"Warning: Volume <%v> is not added to node  <%v> yet. Will retry.",
				volumeName,
				nodeName)

			return false, nil
		},
	)

	if err != nil {
		t.Fatalf(
			"Volume <%v> is not added to node  <%v>. %v",
			volumeName,
			nodeName, err)
	}
}

func waitForVolumeRemovedFromNode(
	t *testing.T,
	volumeName v1.UniqueVolumeName,
	nodeName k8stypes.NodeName,
	asw cache.ActualStateOfWorld) {

	err := retryWithExponentialBackOff(
		time.Duration(500*time.Millisecond),
		func() (bool, error) {
			volumes := asw.GetAttachedVolumes()
			exist := false
			for _, volume := range volumes {
				if volume.VolumeName == volumeName && volume.NodeName == nodeName {
					exist = true
				}
			}
			if exist {
				t.Logf(
					"Warning: Volume <%v> is not removed from the node  <%v> yet. Will retry.",
					volumeName,
					nodeName)

				return false, nil
			}
			return true, nil

		},
	)

	if err != nil {
		t.Fatalf(
			"Volume <%v> is not removed from node  <%v>. %v",
			volumeName,
			nodeName, err)
	}
}

func verifyVolumeAttachedToNode(
	t *testing.T,
	volumeName v1.UniqueVolumeName,
	nodeName k8stypes.NodeName,
	expectedAttachState cache.AttachState,
	asw cache.ActualStateOfWorld,
) {
	attachState := asw.GetAttachState(volumeName, nodeName)
	if attachState != expectedAttachState {
		t.Fatalf("Check volume <%v> is attached to node <%v>, got %v, expected %v",
			volumeName,
			nodeName,
			attachState,
			expectedAttachState)
	}
}

func verifyVolumeReportedAsAttachedToNode(
	t *testing.T,
	volumeName v1.UniqueVolumeName,
	nodeName k8stypes.NodeName,
	isAttached bool,
	asw cache.ActualStateOfWorld,
) {
	result := false
	volumes := asw.GetVolumesToReportAttached()
	for _, volume := range volumes[nodeName] {
		if volume.Name == volumeName {
			result = true
		}
	}

	if result == isAttached {
		return
	}
	t.Fatalf("Check volume <%v> is reported as attached to node <%v>, got %v, expected %v",
		volumeName,
		nodeName,
		result,
		isAttached)

}

func verifyVolumeNoStatusUpdateNeeded(
	t *testing.T,
	volumeName v1.UniqueVolumeName,
	nodeName k8stypes.NodeName,
	asw cache.ActualStateOfWorld,
) {
	volumes := asw.GetVolumesToReportAttached()
	for _, volume := range volumes[nodeName] {
		if volume.Name == volumeName {
			t.Fatalf("Check volume <%v> is reported as need to update status on node <%v>, expected false",
				volumeName,
				nodeName)
		}
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
