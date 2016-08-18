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
	"k8s.io/kubernetes/pkg/controller/volume/snapshot/cache"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/snapshot/testing"
	"k8s.io/kubernetes/pkg/util/wait"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

const (
	reconcilerLoopPeriod      time.Duration = 0 * time.Millisecond
	maxWaitForUnmountDuration time.Duration = 50 * time.Millisecond
)

// Calls Run()
// Verifies there are no calls to CreateSnapshot.
func Test_Run_Positive_DoNothing(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	soe := operationexecutor.NewOperationExecutor(
		fakeKubeClient, volumePluginMgr)

	reconciler := NewReconciler(
		reconcilerLoopPeriod, asw, soe)

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Assert
	waitForCreateSnapshotCallCount(t, 0 /* expectedCallCount */, fakePlugin)
	verifyCreateSnapshotCallCount(t, true /* expectZeroCreateSnapshotCallCount */, fakePlugin)
}

// Populates actualStateOfWorld cache with one volume.
// Calls Run()
// Verifies there is one create snapshot call.
func Test_Run_Positive_OneDesiredCreateSnapshot(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	soe := operationexecutor.NewOperationExecutor(
		fakeKubeClient, volumePluginMgr)
	reconciler := NewReconciler(
		reconcilerLoopPeriod, asw, soe)

	volumeName := api.UniqueVolumeName("volume-name")
	volumeSpec := controllervolumetesting.GetTestVolumeSpec(string(volumeName), volumeName)
	pvcName := "mypvc"
	pvcNamespace := "mynamespace"
	pvc := controllervolumetesting.GetTestPvc(pvcName, pvcNamespace, string(volumeName))
	snapshotName := "snapshot-name"
	_, err := asw.AddVolume(volumeSpec, pvc, snapshotName)
	if err != nil {
		t.Fatalf("AddVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Assert
	waitForCreateSnapshotCallCount(t, 1 /* expectedCallCount */, fakePlugin)
	verifyCreateSnapshotCallCount(t, false /* expectZeroCreateSnapshotCallCount */, fakePlugin)
}

func waitForCreateSnapshotCallCount(
	t *testing.T,
	expectedCallCount int,
	fakePlugin *volumetesting.FakeVolumePlugin) {
	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			actualCallCount := fakePlugin.GetCreateSnapshotCallCount()
			if actualCallCount >= expectedCallCount {
				return true, nil
			}
			t.Logf(
				"Warning: Wrong CreateSnapshotCallCount. Expected: <%v> Actual: <%v>. Will retry.",
				expectedCallCount,
				actualCallCount)
			return false, nil
		},
	)

	if err != nil {
		t.Fatalf(
			"Timed out waiting for CreateSnapshotCallCount. Expected: <%v> Actual: <%v>",
			expectedCallCount,
			fakePlugin.GetCreateSnapshotCallCount())
	}
}

func verifyCreateSnapshotCallCount(
	t *testing.T,
	expectZeroCreateSnapshotCallCount bool,
	fakePlugin *volumetesting.FakeVolumePlugin) {

	if expectZeroCreateSnapshotCallCount &&
		fakePlugin.GetCreateSnapshotCallCount() != 0 {
		t.Fatalf(
			"Wrong CreateSnapshotCallCount. Expected: <0> Actual: <%v>",
			fakePlugin.GetCreateSnapshotCallCount())
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
