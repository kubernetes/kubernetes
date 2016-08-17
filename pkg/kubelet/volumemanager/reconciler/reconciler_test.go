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
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

const (
	// reconcilerLoopSleepDuration is the amount of time the reconciler loop
	// waits between successive executions
	reconcilerLoopSleepDuration      time.Duration = 0 * time.Millisecond
	reconcilerReconstructSleepPeriod time.Duration = 10 * time.Minute
	// waitForAttachTimeout is the maximum amount of time a
	// operationexecutor.Mount call will wait for a volume to be attached.
	waitForAttachTimeout time.Duration = 1 * time.Second
	nodeName             string        = "myhostname"
	kubeletPodsDir       string        = "fake-dir"
)

// Calls Run()
// Verifies there are no calls to attach, detach, mount, unmount, etc.
func Test_Run_Positive_DoNothing(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	oex := operationexecutor.NewOperationExecutor(kubeClient, volumePluginMgr)
	reconciler := NewReconciler(
		kubeClient,
		false, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		reconcilerReconstructSleepPeriod,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		oex,
		&mount.FakeMounter{},
		volumePluginMgr,
		kubeletPodsDir)

	// Act
	runReconciler(reconciler)

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
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	oex := operationexecutor.NewOperationExecutor(kubeClient, volumePluginMgr)
	reconciler := NewReconciler(
		kubeClient,
		false, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		reconcilerReconstructSleepPeriod,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		oex,
		&mount.FakeMounter{},
		volumePluginMgr,
		kubeletPodsDir)
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
	runReconciler(reconciler)
	waitForMount(t, fakePlugin, generatedVolumeName, asw)
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
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	oex := operationexecutor.NewOperationExecutor(kubeClient, volumePluginMgr)
	reconciler := NewReconciler(
		kubeClient,
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		reconcilerReconstructSleepPeriod,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		oex,
		&mount.FakeMounter{},
		volumePluginMgr,
		kubeletPodsDir)
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
	dsw.MarkVolumesReportedInUse([]api.UniqueVolumeName{generatedVolumeName})

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	runReconciler(reconciler)
	waitForMount(t, fakePlugin, generatedVolumeName, asw)

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
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	oex := operationexecutor.NewOperationExecutor(kubeClient, volumePluginMgr)
	reconciler := NewReconciler(
		kubeClient,
		false, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		reconcilerReconstructSleepPeriod,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		oex,
		&mount.FakeMounter{},
		volumePluginMgr,
		kubeletPodsDir)
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
	runReconciler(reconciler)
	waitForMount(t, fakePlugin, generatedVolumeName, asw)
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
	waitForDetach(t, fakePlugin, generatedVolumeName, asw)

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
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	oex := operationexecutor.NewOperationExecutor(kubeClient, volumePluginMgr)
	reconciler := NewReconciler(
		kubeClient,
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		reconcilerReconstructSleepPeriod,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		oex,
		&mount.FakeMounter{},
		volumePluginMgr,
		kubeletPodsDir)
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
	runReconciler(reconciler)

	dsw.MarkVolumesReportedInUse([]api.UniqueVolumeName{generatedVolumeName})
	waitForMount(t, fakePlugin, generatedVolumeName, asw)

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
	waitForDetach(t, fakePlugin, generatedVolumeName, asw)

	// Assert
	assert.NoError(t, volumetesting.VerifyTearDownCallCount(
		1 /* expectedTearDownCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))
}

func waitForMount(
	t *testing.T,
	fakePlugin *volumetesting.FakeVolumePlugin,
	volumeName api.UniqueVolumeName,
	asw cache.ActualStateOfWorld) {
	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			mountedVolumes := asw.GetMountedVolumes()
			for _, mountedVolume := range mountedVolumes {
				if mountedVolume.VolumeName == volumeName {
					return true, nil
				}
			}

			return false, nil
		},
	)

	if err != nil {
		t.Fatalf("Timed out waiting for volume %q to be attached.", volumeName)
	}
}

func waitForDetach(
	t *testing.T,
	fakePlugin *volumetesting.FakeVolumePlugin,
	volumeName api.UniqueVolumeName,
	asw cache.ActualStateOfWorld) {
	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			if asw.VolumeExists(volumeName) {
				return false, nil
			}

			return true, nil
		},
	)

	if err != nil {
		t.Fatalf("Timed out waiting for volume %q to be detached.", volumeName)
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

func createTestClient() *fake.Clientset {
	fakeClient := &fake.Clientset{}
	fakeClient.AddReactor("get", "nodes",
		func(action core.Action) (bool, runtime.Object, error) {
			return true, &api.Node{
				ObjectMeta: api.ObjectMeta{Name: nodeName},
				Status: api.NodeStatus{
					VolumesAttached: []api.AttachedVolume{
						{
							Name:       "fake-plugin/volume-name",
							DevicePath: "fake/path",
						},
					}},
				Spec: api.NodeSpec{ExternalID: nodeName},
			}, nil
		})
	fakeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("no reaction implemented for %s", action)
	})
	return fakeClient
}

func runReconciler(reconciler Reconciler) {
	sourcesReady := config.NewSourcesReady(func(_ sets.String) bool { return false })
	go reconciler.Run(sourcesReady, wait.NeverStop)
}
