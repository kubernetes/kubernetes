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
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

const (
	// reconcilerLoopSleepDuration is the amount of time the reconciler loop
	// waits between successive executions
	reconcilerLoopSleepDuration time.Duration = 1 * time.Nanosecond
	// waitForAttachTimeout is the maximum amount of time a
	// operationexecutor.Mount call will wait for a volume to be attached.
	waitForAttachTimeout time.Duration     = 1 * time.Second
	nodeName             k8stypes.NodeName = k8stypes.NodeName("mynodename")
	kubeletPodsDir       string            = "fake-dir"
)

func hasAddedPods() bool { return true }

// Calls Run()
// Verifies there are no calls to attach, detach, mount, unmount, etc.
func Test_Run_Positive_DoNothing(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		false, /* checkNodeCapabilitiesBeforeMount */
		fakeHandler,
	))
	reconciler := NewReconciler(
		kubeClient,
		false, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		&mount.FakeMounter{},
		&mount.FakeHostUtil{},
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
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		false, /* checkNodeCapabilitiesBeforeMount */
		fakeHandler))
	reconciler := NewReconciler(
		kubeClient,
		false, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		&mount.FakeMounter{},
		&mount.FakeHostUtil{},
		volumePluginMgr,
		kubeletPodsDir)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := util.GetUniquePodName(pod)
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
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		false, /* checkNodeCapabilitiesBeforeMount */
		fakeHandler))
	reconciler := NewReconciler(
		kubeClient,
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		&mount.FakeMounter{},
		&mount.FakeHostUtil{},
		volumePluginMgr,
		kubeletPodsDir)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := util.GetUniquePodName(pod)
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)
	dsw.MarkVolumesReportedInUse([]v1.UniqueVolumeName{generatedVolumeName})

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
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		false, /* checkNodeCapabilitiesBeforeMount */
		fakeHandler))
	reconciler := NewReconciler(
		kubeClient,
		false, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		&mount.FakeMounter{},
		&mount.FakeHostUtil{},
		volumePluginMgr,
		kubeletPodsDir)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := util.GetUniquePodName(pod)
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
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		false, /* checkNodeCapabilitiesBeforeMount */
		fakeHandler))
	reconciler := NewReconciler(
		kubeClient,
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		&mount.FakeMounter{},
		&mount.FakeHostUtil{},
		volumePluginMgr,
		kubeletPodsDir)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := util.GetUniquePodName(pod)
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	runReconciler(reconciler)

	dsw.MarkVolumesReportedInUse([]v1.UniqueVolumeName{generatedVolumeName})
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

// Populates desiredStateOfWorld cache with one volume/pod.
// Calls Run()
// Verifies there are attach/get map paths/setupDevice calls and
// no detach/teardownDevice calls.
func Test_Run_Positive_VolumeAttachAndMap(t *testing.T) {
	// Enable BlockVolume feature gate
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BlockVolume, true)()

	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		false, /* checkNodeCapabilitiesBeforeMount */
		fakeHandler))
	reconciler := NewReconciler(
		kubeClient,
		false, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		&mount.FakeMounter{},
		&mount.FakeHostUtil{},
		volumePluginMgr,
		kubeletPodsDir)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{},
	}

	mode := v1.PersistentVolumeBlock
	gcepv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{UID: "001", Name: "volume-name"},
		Spec: v1.PersistentVolumeSpec{
			Capacity:               v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G")},
			PersistentVolumeSource: v1.PersistentVolumeSource{GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{PDName: "fake-device1"}},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
			},
			VolumeMode: &mode,
		},
	}

	volumeSpec := &volume.Spec{
		PersistentVolume: gcepv,
	}
	podName := util.GetUniquePodName(pod)
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
	assert.NoError(t, volumetesting.VerifyGetMapDeviceCallCount(
		1 /* expectedGetMapDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroTearDownDeviceCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))
}

// Populates desiredStateOfWorld cache with one volume/pod.
// Enables controllerAttachDetachEnabled.
// Calls Run()
// Verifies there are two get map path calls, a setupDevice call
// and no teardownDevice call.
// Verifies there are no attach/detach calls.
func Test_Run_Positive_BlockVolumeMapControllerAttachEnabled(t *testing.T) {
	// Enable BlockVolume feature gate
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BlockVolume, true)()

	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		false, /* checkNodeCapabilitiesBeforeMount */
		fakeHandler))
	reconciler := NewReconciler(
		kubeClient,
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		&mount.FakeMounter{},
		&mount.FakeHostUtil{},
		volumePluginMgr,
		kubeletPodsDir)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{},
	}

	mode := v1.PersistentVolumeBlock
	gcepv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{UID: "001", Name: "volume-name"},
		Spec: v1.PersistentVolumeSpec{
			Capacity:               v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G")},
			PersistentVolumeSource: v1.PersistentVolumeSource{GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{PDName: "fake-device1"}},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
			},
			VolumeMode: &mode,
		},
	}

	volumeSpec := &volume.Spec{
		PersistentVolume: gcepv,
	}
	podName := util.GetUniquePodName(pod)
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)
	dsw.MarkVolumesReportedInUse([]v1.UniqueVolumeName{generatedVolumeName})

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
	assert.NoError(t, volumetesting.VerifyGetMapDeviceCallCount(
		1 /* expectedGetMapDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroTearDownDeviceCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))
}

// Populates desiredStateOfWorld cache with one volume/pod.
// Calls Run()
// Verifies there is one attach call, two get map path calls,
// setupDevice call and no detach calls.
// Deletes volume/pod from desired state of world.
// Verifies one detach/teardownDevice calls are issued.
func Test_Run_Positive_BlockVolumeAttachMapUnmapDetach(t *testing.T) {
	// Enable BlockVolume feature gate
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BlockVolume, true)()

	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		false, /* checkNodeCapabilitiesBeforeMount */
		fakeHandler))
	reconciler := NewReconciler(
		kubeClient,
		false, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		&mount.FakeMounter{},
		&mount.FakeHostUtil{},
		volumePluginMgr,
		kubeletPodsDir)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{},
	}

	mode := v1.PersistentVolumeBlock
	gcepv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{UID: "001", Name: "volume-name"},
		Spec: v1.PersistentVolumeSpec{
			Capacity:               v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G")},
			PersistentVolumeSource: v1.PersistentVolumeSource{GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{PDName: "fake-device1"}},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
			},
			VolumeMode: &mode,
		},
	}

	volumeSpec := &volume.Spec{
		PersistentVolume: gcepv,
	}
	podName := util.GetUniquePodName(pod)
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
	assert.NoError(t, volumetesting.VerifyGetMapDeviceCallCount(
		1 /* expectedGetMapDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroTearDownDeviceCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))

	// Act
	dsw.DeletePodFromVolume(podName, generatedVolumeName)
	waitForDetach(t, fakePlugin, generatedVolumeName, asw)

	// Assert
	assert.NoError(t, volumetesting.VerifyTearDownDeviceCallCount(
		1 /* expectedTearDownDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyDetachCallCount(
		1 /* expectedDetachCallCount */, fakePlugin))
}

// Populates desiredStateOfWorld cache with one volume/pod.
// Enables controllerAttachDetachEnabled.
// Calls Run()
// Verifies two map path calls are made and no teardownDevice/detach calls.
// Deletes volume/pod from desired state of world.
// Verifies one teardownDevice call is made.
// Verifies there are no attach/detach calls made.
func Test_Run_Positive_VolumeUnmapControllerAttachEnabled(t *testing.T) {
	// Enable BlockVolume feature gate
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BlockVolume, true)()

	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		false, /* checkNodeCapabilitiesBeforeMount */
		fakeHandler))
	reconciler := NewReconciler(
		kubeClient,
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		&mount.FakeMounter{},
		&mount.FakeHostUtil{},
		volumePluginMgr,
		kubeletPodsDir)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{},
	}

	mode := v1.PersistentVolumeBlock
	gcepv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{UID: "001", Name: "volume-name"},
		Spec: v1.PersistentVolumeSpec{
			Capacity:               v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G")},
			PersistentVolumeSource: v1.PersistentVolumeSource{GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{PDName: "fake-device1"}},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
			},
			VolumeMode: &mode,
		},
	}

	volumeSpec := &volume.Spec{
		PersistentVolume: gcepv,
	}
	podName := util.GetUniquePodName(pod)
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	runReconciler(reconciler)

	dsw.MarkVolumesReportedInUse([]v1.UniqueVolumeName{generatedVolumeName})
	waitForMount(t, fakePlugin, generatedVolumeName, asw)

	// Assert
	assert.NoError(t, volumetesting.VerifyZeroAttachCalls(fakePlugin))
	assert.NoError(t, volumetesting.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyGetMapDeviceCallCount(
		1 /* expectedGetMapDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroTearDownDeviceCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))

	// Act
	dsw.DeletePodFromVolume(podName, generatedVolumeName)
	waitForDetach(t, fakePlugin, generatedVolumeName, asw)

	// Assert
	assert.NoError(t, volumetesting.VerifyTearDownDeviceCallCount(
		1 /* expectedTearDownDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))
}

func Test_GenerateMapVolumeFunc_Plugin_Not_Found(t *testing.T) {
	testCases := map[string]struct {
		volumePlugins  []volume.VolumePlugin
		expectErr      bool
		expectedErrMsg string
	}{
		"volumePlugin is nil": {
			volumePlugins:  []volume.VolumePlugin{},
			expectErr:      true,
			expectedErrMsg: "MapVolume.FindMapperPluginBySpec failed",
		},
		"blockVolumePlugin is nil": {
			volumePlugins:  volumetesting.NewFakeFileVolumePlugin(),
			expectErr:      true,
			expectedErrMsg: "MapVolume.FindMapperPluginBySpec failed to find BlockVolumeMapper plugin. Volume plugin is nil.",
		},
	}

	// Enable BlockVolume feature gate
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BlockVolume, true)()

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			volumePluginMgr := &volume.VolumePluginMgr{}
			volumePluginMgr.InitPlugins(tc.volumePlugins, nil, nil)
			asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
			oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
				nil, /* kubeClient */
				volumePluginMgr,
				nil,   /* fakeRecorder */
				false, /* checkNodeCapabilitiesBeforeMount */
				nil))

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					UID:  "pod1uid",
				},
				Spec: v1.PodSpec{},
			}
			volumeMode := v1.PersistentVolumeBlock
			tmpSpec := &volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{VolumeMode: &volumeMode}}}
			volumeToMount := operationexecutor.VolumeToMount{
				Pod:        pod,
				VolumeSpec: tmpSpec}
			err := oex.MountVolume(waitForAttachTimeout, volumeToMount, asw, false)
			// Assert
			if assert.Error(t, err) {
				assert.Contains(t, err.Error(), tc.expectedErrMsg)
			}
		})
	}
}

func Test_GenerateUnmapVolumeFunc_Plugin_Not_Found(t *testing.T) {
	testCases := map[string]struct {
		volumePlugins  []volume.VolumePlugin
		expectErr      bool
		expectedErrMsg string
	}{
		"volumePlugin is nil": {
			volumePlugins:  []volume.VolumePlugin{},
			expectErr:      true,
			expectedErrMsg: "UnmapVolume.FindMapperPluginByName failed",
		},
		"blockVolumePlugin is nil": {
			volumePlugins:  volumetesting.NewFakeFileVolumePlugin(),
			expectErr:      true,
			expectedErrMsg: "UnmapVolume.FindMapperPluginByName failed to find BlockVolumeMapper plugin. Volume plugin is nil.",
		},
	}

	// Enable BlockVolume feature gate
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BlockVolume, true)()

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			volumePluginMgr := &volume.VolumePluginMgr{}
			volumePluginMgr.InitPlugins(tc.volumePlugins, nil, nil)
			asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
			oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
				nil, /* kubeClient */
				volumePluginMgr,
				nil,   /* fakeRecorder */
				false, /* checkNodeCapabilitiesBeforeMount */
				nil))
			volumeMode := v1.PersistentVolumeBlock
			tmpSpec := &volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{VolumeMode: &volumeMode}}}
			volumeToUnmount := operationexecutor.MountedVolume{
				PluginName: "fake-file-plugin",
				VolumeSpec: tmpSpec}
			err := oex.UnmountVolume(volumeToUnmount, asw, "" /* podsDir */)
			// Assert
			if assert.Error(t, err) {
				assert.Contains(t, err.Error(), tc.expectedErrMsg)
			}
		})
	}
}

func Test_GenerateUnmapDeviceFunc_Plugin_Not_Found(t *testing.T) {
	testCases := map[string]struct {
		volumePlugins  []volume.VolumePlugin
		expectErr      bool
		expectedErrMsg string
	}{
		"volumePlugin is nil": {
			volumePlugins:  []volume.VolumePlugin{},
			expectErr:      true,
			expectedErrMsg: "UnmapDevice.FindMapperPluginByName failed",
		},
		"blockVolumePlugin is nil": {
			volumePlugins:  volumetesting.NewFakeFileVolumePlugin(),
			expectErr:      true,
			expectedErrMsg: "UnmapDevice.FindMapperPluginByName failed to find BlockVolumeMapper plugin. Volume plugin is nil.",
		},
	}

	// Enable BlockVolume feature gate
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BlockVolume, true)()

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			volumePluginMgr := &volume.VolumePluginMgr{}
			volumePluginMgr.InitPlugins(tc.volumePlugins, nil, nil)
			asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
			oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
				nil, /* kubeClient */
				volumePluginMgr,
				nil,   /* fakeRecorder */
				false, /* checkNodeCapabilitiesBeforeMount */
				nil))
			var hostutil mount.HostUtils
			volumeMode := v1.PersistentVolumeBlock
			tmpSpec := &volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{VolumeMode: &volumeMode}}}
			deviceToDetach := operationexecutor.AttachedVolume{VolumeSpec: tmpSpec, PluginName: "fake-file-plugin"}
			err := oex.UnmountDevice(deviceToDetach, asw, hostutil)
			// Assert
			if assert.Error(t, err) {
				assert.Contains(t, err.Error(), tc.expectedErrMsg)
			}
		})
	}
}

// Populates desiredStateOfWorld cache with one volume/pod.
// Enables controllerAttachDetachEnabled.
// Calls Run()
// Wait for volume mounted.
// Mark volume as fsResizeRequired in ASW.
// Verifies volume's fsResizeRequired flag is cleared later.
func Test_Run_Positive_VolumeFSResizeControllerAttachEnabled(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExpandInUsePersistentVolumes, true)()

	fs := v1.PersistentVolumeFilesystem
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pv",
			UID:  "pvuid",
		},
		Spec: v1.PersistentVolumeSpec{
			ClaimRef:   &v1.ObjectReference{Name: "pvc"},
			VolumeMode: &fs,
		},
	}
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pvc",
			UID:  "pvcuid",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeName: "pv",
			VolumeMode: &fs,
		},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: pvc.Name,
						},
					},
				},
			},
		},
	}

	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createtestClientWithPVPVC(pv, pvc)
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		false, /* checkNodeCapabilitiesBeforeMount */
		fakeHandler))

	reconciler := NewReconciler(
		kubeClient,
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		&mount.FakeMounter{},
		&mount.FakeHostUtil{},
		volumePluginMgr,
		kubeletPodsDir)

	volumeSpec := &volume.Spec{PersistentVolume: pv}
	podName := util.GetUniquePodName(pod)
	volumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)
	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}
	dsw.MarkVolumesReportedInUse([]v1.UniqueVolumeName{volumeName})

	// Start the reconciler to fill ASW.
	stopChan, stoppedChan := make(chan struct{}), make(chan struct{})
	go func() {
		reconciler.Run(stopChan)
		close(stoppedChan)
	}()
	waitForMount(t, fakePlugin, volumeName, asw)
	// Stop the reconciler.
	close(stopChan)
	<-stoppedChan

	// Mark volume as fsResizeRequired.
	asw.MarkFSResizeRequired(volumeName, podName)
	_, _, podExistErr := asw.PodExistsInVolume(podName, volumeName)
	if !cache.IsFSResizeRequiredError(podExistErr) {
		t.Fatalf("Volume should be marked as fsResizeRequired, but receive unexpected error: %v", podExistErr)
	}

	// Start the reconciler again, we hope reconciler will perform the
	// resize operation and clear the fsResizeRequired flag for volume.
	go reconciler.Run(wait.NeverStop)

	waitErr := retryWithExponentialBackOff(500*time.Millisecond, func() (done bool, err error) {
		mounted, _, err := asw.PodExistsInVolume(podName, volumeName)
		return mounted && err == nil, nil
	})
	if waitErr != nil {
		t.Fatal("Volume resize should succeeded")
	}
}

func waitForMount(
	t *testing.T,
	fakePlugin *volumetesting.FakeVolumePlugin,
	volumeName v1.UniqueVolumeName,
	asw cache.ActualStateOfWorld) {
	err := retryWithExponentialBackOff(
		time.Duration(500*time.Millisecond),
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
	volumeName v1.UniqueVolumeName,
	asw cache.ActualStateOfWorld) {
	err := retryWithExponentialBackOff(
		time.Duration(500*time.Millisecond),
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
			return true, &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: string(nodeName)},
				Status: v1.NodeStatus{
					VolumesAttached: []v1.AttachedVolume{
						{
							Name:       "fake-plugin/fake-device1",
							DevicePath: "/fake/path",
						},
					}},
			}, nil
		})
	fakeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("no reaction implemented for %s", action)
	})
	return fakeClient
}

func runReconciler(reconciler Reconciler) {
	go reconciler.Run(wait.NeverStop)
}

func createtestClientWithPVPVC(pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) *fake.Clientset {
	fakeClient := &fake.Clientset{}
	fakeClient.AddReactor("get", "nodes",
		func(action core.Action) (bool, runtime.Object, error) {
			return true, &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: string(nodeName)},
				Status: v1.NodeStatus{
					VolumesAttached: []v1.AttachedVolume{
						{
							Name:       "fake-plugin/pv",
							DevicePath: "fake/path",
						},
					}},
			}, nil
		})
	fakeClient.AddReactor("get", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
		return true, pvc, nil
	})
	fakeClient.AddReactor("get", "persistentvolumes", func(action core.Action) (bool, runtime.Object, error) {
		return true, pv, nil
	})
	fakeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("no reaction implemented for %s", action)
	})
	return fakeClient
}

func Test_Run_Positive_VolumeMountControllerAttachEnabledRace(t *testing.T) {
	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)

	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		false, /* checkNodeCapabilitiesBeforeMount */
		fakeHandler))
	reconciler := NewReconciler(
		kubeClient,
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		&mount.FakeMounter{},
		&mount.FakeHostUtil{},
		volumePluginMgr,
		kubeletPodsDir)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	// Some steps are executes out of order in callbacks, follow the numbers.

	// 1. Add a volume to DSW and wait until it's mounted
	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := util.GetUniquePodName(pod)
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)
	dsw.MarkVolumesReportedInUse([]v1.UniqueVolumeName{generatedVolumeName})

	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}
	// Start the reconciler to fill ASW.
	stopChan, stoppedChan := make(chan struct{}), make(chan struct{})
	go func() {
		reconciler.Run(stopChan)
		close(stoppedChan)
	}()
	waitForMount(t, fakePlugin, generatedVolumeName, asw)
	// Stop the reconciler.
	close(stopChan)
	<-stoppedChan

	finished := make(chan interface{})
	fakePlugin.UnmountDeviceHook = func(mountPath string) error {
		// Act:
		// 3. While a volume is being unmounted, add it back to the desired state of world
		klog.Infof("UnmountDevice called")
		generatedVolumeName, err = dsw.AddPodToVolume(
			podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */)
		dsw.MarkVolumesReportedInUse([]v1.UniqueVolumeName{generatedVolumeName})
		return nil
	}

	fakePlugin.WaitForAttachHook = func(spec *volume.Spec, devicePath string, pod *v1.Pod, spectimeout time.Duration) (string, error) {
		// Assert
		// 4. When the volume is mounted again, expect that UnmountDevice operation did not clear devicePath
		if devicePath == "" {
			t.Errorf("Expected WaitForAttach called with devicePath from Node.Status")
			close(finished)
			return "", fmt.Errorf("Expected devicePath from Node.Status")
		}
		close(finished)
		return devicePath, nil
	}

	// Start the reconciler again.
	go reconciler.Run(wait.NeverStop)

	// 2. Delete the volume from DSW (and wait for callbacks)
	dsw.DeletePodFromVolume(podName, generatedVolumeName)

	<-finished
	waitForMount(t, fakePlugin, generatedVolumeName, asw)
}
