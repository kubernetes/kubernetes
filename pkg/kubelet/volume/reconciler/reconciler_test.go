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

package reconciler

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/volume/cache"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

const (
	// reconcilerLoopSleepDuration is the amount of time the reconciler loop
	// waits between successive executions
	reconcilerLoopSleepDuration time.Duration = 0 * time.Millisecond

	// waitForAttachTimeout is the maximum amount of time a
	// operationexecutor.Mount call will wait for a volume to be attached.
	waitForAttachTimeout time.Duration = 1 * time.Second
)

// Calls Run()
// Verifies there are no calls to attach, detach, mount, unmount, etc.
func Test_Run_Positive_DoNothing(t *testing.T) {
	// Arrange
	nodeName := "myhostname"
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	oex := operationexecutor.NewOperationExecutor(volumePluginMgr)
	reconciler := NewReconciler(
		false, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		oex)

	// Act
	go reconciler.Run(wait.NeverStop)

	// Assert
	assert.NoError(t, volumetesting.VerifyZeroAttachCalls(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroWaitForAttachCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroMountDeviceCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroSetUpCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroTearDownCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))
}

// Populates desiredStateOfWorld cache with one volume/pod.
// Calls Run()
// Verifies there is are attach/mount/etc calls and no detach/unmount calls.
func Test_Run_Positive_VolumeAttachAndMount(t *testing.T) {
	// Arrange
	nodeName := "myhostname"
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	oex := operationexecutor.NewOperationExecutor(volumePluginMgr)
	reconciler := NewReconciler(
		false, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		oex)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "volume-name",
					VolumeSource: api.VolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := volumehelper.GetUniquePodName(pod)
	_, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	go reconciler.Run(wait.NeverStop)
	waitForAttach(t, fakePlugin, asw)

	// Assert
	assert.NoError(t, volumetesting.VerifyAttachCallCount(
		1 /* expectedAttachCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyMountDeviceCallCount(
		1 /* expectedMountDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifySetUpCallCount(
		1 /* expectedSetUpCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroTearDownCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))
}

// Populates desiredStateOfWorld cache with one volume/pod.
// Enables controllerAttachDetachEnabled.
// Calls Run()
// Verifies there is one mount call and no unmount calls.
// Verifies there are no attach/detach calls.
func Test_Run_Positive_VolumeMountControllerAttachEnabled(t *testing.T) {
	// Arrange
	nodeName := "myhostname"
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	oex := operationexecutor.NewOperationExecutor(volumePluginMgr)
	reconciler := NewReconciler(
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		oex)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "volume-name",
					VolumeSource: api.VolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := volumehelper.GetUniquePodName(pod)
	_, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	go reconciler.Run(wait.NeverStop)
	waitForAttach(t, fakePlugin, asw)

	// Assert
	assert.NoError(t, volumetesting.VerifyZeroAttachCalls(fakePlugin))
	assert.NoError(t, volumetesting.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyMountDeviceCallCount(
		1 /* expectedMountDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifySetUpCallCount(
		1 /* expectedSetUpCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroTearDownCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))
}

// Populates desiredStateOfWorld cache with one volume/pod.
// Calls Run()
// Verifies there is one attach/mount/etc call and no detach calls.
// Deletes volume/pod from desired state of world.
// Verifies detach/unmount calls are issued.
func Test_Run_Positive_VolumeAttachMountUnmountDetach(t *testing.T) {
	// Arrange
	nodeName := "myhostname"
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	oex := operationexecutor.NewOperationExecutor(volumePluginMgr)
	reconciler := NewReconciler(
		false, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		oex)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "volume-name",
					VolumeSource: api.VolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := volumehelper.GetUniquePodName(pod)
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	go reconciler.Run(wait.NeverStop)
	waitForAttach(t, fakePlugin, asw)

	// Assert
	assert.NoError(t, volumetesting.VerifyAttachCallCount(
		1 /* expectedAttachCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyMountDeviceCallCount(
		1 /* expectedMountDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifySetUpCallCount(
		1 /* expectedSetUpCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroTearDownCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))

	// Act
	dsw.DeletePodFromVolume(podName, generatedVolumeName)
	waitForDetach(t, fakePlugin, asw)

	// Assert
	assert.NoError(t, volumetesting.VerifyTearDownCallCount(
		1 /* expectedTearDownCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyDetachCallCount(
		1 /* expectedDetachCallCount */, fakePlugin))
}

// Populates desiredStateOfWorld cache with one volume/pod.
// Enables controllerAttachDetachEnabled.
// Calls Run()
// Verifies one mount call is made and no unmount calls.
// Deletes volume/pod from desired state of world.
// Verifies one unmount call is made.
// Verifies there are no attach/detach calls made.
func Test_Run_Positive_VolumeUnmountControllerAttachEnabled(t *testing.T) {
	// Arrange
	nodeName := "myhostname"
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	oex := operationexecutor.NewOperationExecutor(volumePluginMgr)
	reconciler := NewReconciler(
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		oex)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "volume-name",
					VolumeSource: api.VolumeSource{
						GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := volumehelper.GetUniquePodName(pod)
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	go reconciler.Run(wait.NeverStop)
	waitForAttach(t, fakePlugin, asw)

	// Assert
	assert.NoError(t, volumetesting.VerifyZeroAttachCalls(fakePlugin))
	assert.NoError(t, volumetesting.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyMountDeviceCallCount(
		1 /* expectedMountDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifySetUpCallCount(
		1 /* expectedSetUpCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroTearDownCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))

	// Act
	dsw.DeletePodFromVolume(podName, generatedVolumeName)
	waitForDetach(t, fakePlugin, asw)

	// Assert
	assert.NoError(t, volumetesting.VerifyTearDownCallCount(
		1 /* expectedTearDownCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))
}

func waitForAttach(
	t *testing.T,
	fakePlugin *volumetesting.FakeVolumePlugin,
	asw cache.ActualStateOfWorld) {
	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			mountedVolumes := asw.GetMountedVolumes()
			if len(mountedVolumes) > 0 {
				return true, nil
			}

			return false, nil
		},
	)

	if err != nil {
		t.Fatalf("Timed out waiting for len of asw.GetMountedVolumes() to become non-zero.")
	}
}

func waitForDetach(
	t *testing.T,
	fakePlugin *volumetesting.FakeVolumePlugin,
	asw cache.ActualStateOfWorld) {
	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			attachedVolumes := asw.GetAttachedVolumes()
			if len(attachedVolumes) == 0 {
				return true, nil
			}

			return false, nil
		},
	)

	if err != nil {
		t.Fatalf("Timed out waiting for len of asw.attachedVolumes() to become zero.")
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
