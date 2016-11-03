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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/statusupdater"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
	k8stypes "k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"
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
	ad := operationexecutor.NewOperationExecutor(
		fakeKubeClient, volumePluginMgr, fakeRecorder, false)
	nodeInformer := informers.NewNodeInformer(
		fakeKubeClient, resyncPeriod)
	nsu := statusupdater.NewNodeStatusUpdater(
		fakeKubeClient, nodeInformer, asw)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, dsw, asw, ad, nsu)

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
	ad := operationexecutor.NewOperationExecutor(fakeKubeClient, volumePluginMgr, fakeRecorder, false)
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, dsw, asw, ad, nsu)
	podName := "pod-uid"
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName)
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
	ad := operationexecutor.NewOperationExecutor(fakeKubeClient, volumePluginMgr, fakeRecorder, false)
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, dsw, asw, ad, nsu)
	podName := "pod-uid"
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName)
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
	ad := operationexecutor.NewOperationExecutor(fakeKubeClient, volumePluginMgr, fakeRecorder, false)
	nsu := statusupdater.NewFakeNodeStatusUpdater(false /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, dsw, asw, ad, nsu)
	podName := "pod-uid"
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName)
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
	ad := operationexecutor.NewOperationExecutor(fakeKubeClient, volumePluginMgr, fakeRecorder, false)
	nsu := statusupdater.NewFakeNodeStatusUpdater(true /* returnError */)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, maxWaitForUnmountDuration, syncLoopPeriod, dsw, asw, ad, nsu)
	podName := "pod-uid"
	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	nodeName := k8stypes.NodeName("node-name")
	dsw.AddNode(nodeName)
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
	go reconciler.Run(wait.NeverStop)

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
