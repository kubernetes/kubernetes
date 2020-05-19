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
	"k8s.io/klog/v2"
	"k8s.io/utils/mount"

	v1 "k8s.io/api/core/v1"
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
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

const (
	// reconcilerLoopSleepDuration is the amount of time the reconciler loop
	// waits between successive executions
	reconcilerLoopSleepDuration time.Duration = 1 * time.Nanosecond
	// waitForAttachTimeout is the maximum amount of time a
	// operationexecutor.Mount call will wait for a volume to be attached.
	waitForAttachTimeout         time.Duration     = 1 * time.Second
	nodeName                     k8stypes.NodeName = k8stypes.NodeName("mynodename")
	kubeletPodsDir               string            = "fake-dir"
	testOperationBackOffDuration time.Duration     = 100 * time.Millisecond
	reconcilerSyncWaitDuration   time.Duration     = 10 * time.Second
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
		mount.NewFakeMounter(nil),
		hostutil.NewFakeHostUtil(nil),
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
		mount.NewFakeMounter(nil),
		hostutil.NewFakeHostUtil(nil),
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
		mount.NewFakeMounter(nil),
		hostutil.NewFakeHostUtil(nil),
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
		mount.NewFakeMounter(nil),
		hostutil.NewFakeHostUtil(nil),
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
	waitForDetach(t, generatedVolumeName, asw)

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
		mount.NewFakeMounter(nil),
		hostutil.NewFakeHostUtil(nil),
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
	waitForDetach(t, generatedVolumeName, asw)

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
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			UID:       "pod1uid",
			Namespace: "ns",
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
			ClaimRef:   &v1.ObjectReference{Namespace: "ns", Name: "pvc-volume-name"},
		},
	}

	gcepvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{UID: "pvc-001", Name: "pvc-volume-name", Namespace: "ns"},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeName: "volume-name",
			VolumeMode: &mode,
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase:    v1.ClaimBound,
			Capacity: gcepv.Spec.Capacity,
		},
	}

	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createtestClientWithPVPVC(gcepv, gcepvc)
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
		mount.NewFakeMounter(nil),
		hostutil.NewFakeHostUtil(nil),
		volumePluginMgr,
		kubeletPodsDir)

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
	assert.NoError(t, volumetesting.VerifyGetMapPodDeviceCallCount(
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
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			UID:       "pod1uid",
			Namespace: "ns",
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
			ClaimRef:   &v1.ObjectReference{Namespace: "ns", Name: "pvc-volume-name"},
		},
	}
	gcepvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{UID: "pvc-001", Name: "pvc-volume-name", Namespace: "ns"},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeName: "volume-name",
			VolumeMode: &mode,
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase:    v1.ClaimBound,
			Capacity: gcepv.Spec.Capacity,
		},
	}

	volumeSpec := &volume.Spec{
		PersistentVolume: gcepv,
	}

	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createtestClientWithPVPVC(gcepv, gcepvc, v1.AttachedVolume{
		Name:       "fake-plugin/fake-device1",
		DevicePath: "/fake/path",
	})
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
		mount.NewFakeMounter(nil),
		hostutil.NewFakeHostUtil(nil),
		volumePluginMgr,
		kubeletPodsDir)

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
	assert.NoError(t, volumetesting.VerifyGetMapPodDeviceCallCount(
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
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			UID:       "pod1uid",
			Namespace: "ns",
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
			ClaimRef:   &v1.ObjectReference{Namespace: "ns", Name: "pvc-volume-name"},
		},
	}
	gcepvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{UID: "pvc-001", Name: "pvc-volume-name", Namespace: "ns"},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeName: "volume-name",
			VolumeMode: &mode,
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase:    v1.ClaimBound,
			Capacity: gcepv.Spec.Capacity,
		},
	}

	volumeSpec := &volume.Spec{
		PersistentVolume: gcepv,
	}

	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createtestClientWithPVPVC(gcepv, gcepvc)
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
		mount.NewFakeMounter(nil),
		hostutil.NewFakeHostUtil(nil),
		volumePluginMgr,
		kubeletPodsDir)

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
	assert.NoError(t, volumetesting.VerifyGetMapPodDeviceCallCount(
		1 /* expectedGetMapDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroTearDownDeviceCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))

	// Act
	dsw.DeletePodFromVolume(podName, generatedVolumeName)
	waitForDetach(t, generatedVolumeName, asw)

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
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			UID:       "pod1uid",
			Namespace: "ns",
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
			ClaimRef:   &v1.ObjectReference{Namespace: "ns", Name: "pvc-volume-name"},
		},
	}
	gcepvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{UID: "pvc-001", Name: "pvc-volume-name", Namespace: "ns"},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeName: "volume-name",
			VolumeMode: &mode,
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase:    v1.ClaimBound,
			Capacity: gcepv.Spec.Capacity,
		},
	}

	volumeSpec := &volume.Spec{
		PersistentVolume: gcepv,
	}

	// Arrange
	volumePluginMgr, fakePlugin := volumetesting.GetTestVolumePluginMgr(t)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createtestClientWithPVPVC(gcepv, gcepvc, v1.AttachedVolume{
		Name:       "fake-plugin/fake-device1",
		DevicePath: "/fake/path",
	})
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
		mount.NewFakeMounter(nil),
		hostutil.NewFakeHostUtil(nil),
		volumePluginMgr,
		kubeletPodsDir)

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
	assert.NoError(t, volumetesting.VerifyGetMapPodDeviceCallCount(
		1 /* expectedGetMapDeviceCallCount */, fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroTearDownDeviceCallCount(fakePlugin))
	assert.NoError(t, volumetesting.VerifyZeroDetachCallCount(fakePlugin))

	// Act
	dsw.DeletePodFromVolume(podName, generatedVolumeName)
	waitForDetach(t, generatedVolumeName, asw)

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
			var hostutil hostutil.HostUtils
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
	blockMode := v1.PersistentVolumeBlock
	fsMode := v1.PersistentVolumeFilesystem

	var tests = []struct {
		name       string
		volumeMode *v1.PersistentVolumeMode
	}{
		{
			name:       "expand-fs-volume",
			volumeMode: &fsMode,
		},
		{
			name:       "expand-raw-block",
			volumeMode: &blockMode,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pv := &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pv",
					UID:  "pvuid",
				},
				Spec: v1.PersistentVolumeSpec{
					ClaimRef:   &v1.ObjectReference{Name: "pvc"},
					VolumeMode: tc.volumeMode,
				},
			}
			pvc := &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pvc",
					UID:  "pvcuid",
				},
				Spec: v1.PersistentVolumeClaimSpec{
					VolumeName: "pv",
					VolumeMode: tc.volumeMode,
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
				mount.NewFakeMounter(nil),
				hostutil.NewFakeHostUtil(nil),
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

			waitErr := retryWithExponentialBackOff(testOperationBackOffDuration, func() (done bool, err error) {
				mounted, _, err := asw.PodExistsInVolume(podName, volumeName)
				return mounted && err == nil, nil
			})
			if waitErr != nil {
				t.Fatal("Volume resize should succeeded")
			}
		})
	}
}

func Test_UncertainDeviceGlobalMounts(t *testing.T) {
	var tests = []struct {
		name                   string
		deviceState            operationexecutor.DeviceMountState
		unmountDeviceCallCount int
		volumeName             string
		supportRemount         bool
	}{
		{
			name:                   "timed out operations should result in device marked as uncertain",
			deviceState:            operationexecutor.DeviceMountUncertain,
			unmountDeviceCallCount: 1,
			volumeName:             volumetesting.TimeoutOnMountDeviceVolumeName,
		},
		{
			name:                   "failed operation should result in not-mounted device",
			deviceState:            operationexecutor.DeviceNotMounted,
			unmountDeviceCallCount: 0,
			volumeName:             volumetesting.FailMountDeviceVolumeName,
		},
		{
			name:                   "timeout followed by failed operation should result in non-mounted device",
			deviceState:            operationexecutor.DeviceNotMounted,
			unmountDeviceCallCount: 0,
			volumeName:             volumetesting.TimeoutAndFailOnMountDeviceVolumeName,
		},
		{
			name:                   "success followed by timeout operation should result in mounted device",
			deviceState:            operationexecutor.DeviceGloballyMounted,
			unmountDeviceCallCount: 1,
			volumeName:             volumetesting.SuccessAndTimeoutDeviceName,
			supportRemount:         true,
		},
		{
			name:                   "success followed by failed operation should result in mounted device",
			deviceState:            operationexecutor.DeviceGloballyMounted,
			unmountDeviceCallCount: 1,
			volumeName:             volumetesting.SuccessAndFailOnMountDeviceName,
			supportRemount:         true,
		},
	}

	for _, mode := range []v1.PersistentVolumeMode{v1.PersistentVolumeBlock, v1.PersistentVolumeFilesystem} {
		for _, tc := range tests {
			testName := fmt.Sprintf("%s [%s]", tc.name, mode)
			t.Run(testName+"[", func(t *testing.T) {

				pv := &v1.PersistentVolume{
					ObjectMeta: metav1.ObjectMeta{
						Name: tc.volumeName,
						UID:  "pvuid",
					},
					Spec: v1.PersistentVolumeSpec{
						ClaimRef:   &v1.ObjectReference{Name: "pvc"},
						VolumeMode: &mode,
					},
				}
				pvc := &v1.PersistentVolumeClaim{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pvc",
						UID:  "pvcuid",
					},
					Spec: v1.PersistentVolumeClaimSpec{
						VolumeName: tc.volumeName,
						VolumeMode: &mode,
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
				fakePlugin.SupportsRemount = tc.supportRemount

				dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
				asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
				kubeClient := createtestClientWithPVPVC(pv, pvc, v1.AttachedVolume{
					Name:       v1.UniqueVolumeName(fmt.Sprintf("fake-plugin/%s", tc.volumeName)),
					DevicePath: "fake/path",
				})
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
					hostutil.NewFakeHostUtil(nil),
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
				waitForVolumeToExistInASW(t, volumeName, asw)
				if tc.volumeName == volumetesting.TimeoutAndFailOnMountDeviceVolumeName {
					// Wait upto 10s for reconciler to catch up
					time.Sleep(reconcilerSyncWaitDuration)
				}

				if tc.volumeName == volumetesting.SuccessAndFailOnMountDeviceName ||
					tc.volumeName == volumetesting.SuccessAndTimeoutDeviceName {
					// wait for mount and then break it via remount
					waitForMount(t, fakePlugin, volumeName, asw)
					asw.MarkRemountRequired(podName)
					time.Sleep(reconcilerSyncWaitDuration)
				}

				if tc.deviceState == operationexecutor.DeviceMountUncertain {
					waitForUncertainGlobalMount(t, volumeName, asw)
				}

				if tc.deviceState == operationexecutor.DeviceGloballyMounted {
					waitForMount(t, fakePlugin, volumeName, asw)
				}

				dsw.DeletePodFromVolume(podName, volumeName)
				waitForDetach(t, volumeName, asw)
				if mode == v1.PersistentVolumeFilesystem {
					err = volumetesting.VerifyUnmountDeviceCallCount(tc.unmountDeviceCallCount, fakePlugin)
				} else {
					if tc.unmountDeviceCallCount == 0 {
						err = volumetesting.VerifyZeroTearDownDeviceCallCount(fakePlugin)
					} else {
						err = volumetesting.VerifyTearDownDeviceCallCount(tc.unmountDeviceCallCount, fakePlugin)
					}
				}
				if err != nil {
					t.Errorf("Error verifying UnMountDeviceCallCount: %v", err)
				}
			})
		}
	}
}

func Test_UncertainVolumeMountState(t *testing.T) {
	var tests = []struct {
		name                   string
		volumeState            operationexecutor.VolumeMountState
		unmountDeviceCallCount int
		unmountVolumeCount     int
		volumeName             string
		supportRemount         bool
	}{
		{
			name:                   "timed out operations should result in volume marked as uncertain",
			volumeState:            operationexecutor.VolumeMountUncertain,
			unmountDeviceCallCount: 1,
			unmountVolumeCount:     1,
			volumeName:             volumetesting.TimeoutOnSetupVolumeName,
		},
		{
			name:                   "failed operation should result in not-mounted volume",
			volumeState:            operationexecutor.VolumeNotMounted,
			unmountDeviceCallCount: 1,
			unmountVolumeCount:     0,
			volumeName:             volumetesting.FailOnSetupVolumeName,
		},
		{
			name:                   "timeout followed by failed operation should result in non-mounted volume",
			volumeState:            operationexecutor.VolumeNotMounted,
			unmountDeviceCallCount: 1,
			unmountVolumeCount:     0,
			volumeName:             volumetesting.TimeoutAndFailOnSetupVolumeName,
		},
		{
			name:                   "success followed by timeout operation should result in mounted volume",
			volumeState:            operationexecutor.VolumeMounted,
			unmountDeviceCallCount: 1,
			unmountVolumeCount:     1,
			volumeName:             volumetesting.SuccessAndTimeoutSetupVolumeName,
			supportRemount:         true,
		},
		{
			name:                   "success followed by failed operation should result in mounted volume",
			volumeState:            operationexecutor.VolumeMounted,
			unmountDeviceCallCount: 1,
			unmountVolumeCount:     1,
			volumeName:             volumetesting.SuccessAndFailOnSetupVolumeName,
			supportRemount:         true,
		},
	}

	for _, mode := range []v1.PersistentVolumeMode{v1.PersistentVolumeBlock, v1.PersistentVolumeFilesystem} {
		for _, tc := range tests {
			testName := fmt.Sprintf("%s [%s]", tc.name, mode)
			t.Run(testName, func(t *testing.T) {
				pv := &v1.PersistentVolume{
					ObjectMeta: metav1.ObjectMeta{
						Name: tc.volumeName,
						UID:  "pvuid",
					},
					Spec: v1.PersistentVolumeSpec{
						ClaimRef:   &v1.ObjectReference{Name: "pvc"},
						VolumeMode: &mode,
					},
				}
				pvc := &v1.PersistentVolumeClaim{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pvc",
						UID:  "pvcuid",
					},
					Spec: v1.PersistentVolumeClaimSpec{
						VolumeName: tc.volumeName,
						VolumeMode: &mode,
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
				fakePlugin.SupportsRemount = tc.supportRemount
				dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
				asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
				kubeClient := createtestClientWithPVPVC(pv, pvc, v1.AttachedVolume{
					Name:       v1.UniqueVolumeName(fmt.Sprintf("fake-plugin/%s", tc.volumeName)),
					DevicePath: "fake/path",
				})
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
					hostutil.NewFakeHostUtil(nil),
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
				waitForVolumeToExistInASW(t, volumeName, asw)
				if tc.volumeName == volumetesting.TimeoutAndFailOnSetupVolumeName {
					// Wait upto 10s for reconciler to catchup
					time.Sleep(reconcilerSyncWaitDuration)
				}

				if tc.volumeName == volumetesting.SuccessAndFailOnSetupVolumeName ||
					tc.volumeName == volumetesting.SuccessAndTimeoutSetupVolumeName {
					// wait for mount and then break it via remount
					waitForMount(t, fakePlugin, volumeName, asw)
					asw.MarkRemountRequired(podName)
					time.Sleep(reconcilerSyncWaitDuration)
				}

				if tc.volumeState == operationexecutor.VolumeMountUncertain {
					waitForUncertainPodMount(t, volumeName, asw)
				}

				if tc.volumeState == operationexecutor.VolumeMounted {
					waitForMount(t, fakePlugin, volumeName, asw)
				}

				dsw.DeletePodFromVolume(podName, volumeName)
				waitForDetach(t, volumeName, asw)

				if mode == v1.PersistentVolumeFilesystem {
					if err := volumetesting.VerifyUnmountDeviceCallCount(tc.unmountDeviceCallCount, fakePlugin); err != nil {
						t.Errorf("Error verifying UnMountDeviceCallCount: %v", err)
					}
					if err := volumetesting.VerifyTearDownCallCount(tc.unmountVolumeCount, fakePlugin); err != nil {
						t.Errorf("Error verifying UnMountDeviceCallCount: %v", err)
					}
				} else {
					if tc.unmountVolumeCount == 0 {
						if err := volumetesting.VerifyZeroUnmapPodDeviceCallCount(fakePlugin); err != nil {
							t.Errorf("Error verifying UnMountDeviceCallCount: %v", err)
						}
					} else {
						if err := volumetesting.VerifyUnmapPodDeviceCallCount(tc.unmountVolumeCount, fakePlugin); err != nil {
							t.Errorf("Error verifying UnMountDeviceCallCount: %v", err)
						}
					}
					if tc.unmountDeviceCallCount == 0 {
						if err := volumetesting.VerifyZeroTearDownDeviceCallCount(fakePlugin); err != nil {
							t.Errorf("Error verifying UnMountDeviceCallCount: %v", err)
						}
					} else {
						if err := volumetesting.VerifyTearDownDeviceCallCount(tc.unmountDeviceCallCount, fakePlugin); err != nil {
							t.Errorf("Error verifying UnMountDeviceCallCount: %v", err)
						}
					}
				}
			})
		}
	}
}

func waitForUncertainGlobalMount(t *testing.T, volumeName v1.UniqueVolumeName, asw cache.ActualStateOfWorld) {
	// check if volume is globally mounted in uncertain state
	err := retryWithExponentialBackOff(
		testOperationBackOffDuration,
		func() (bool, error) {
			unmountedVolumes := asw.GetUnmountedVolumes()
			for _, v := range unmountedVolumes {
				if v.VolumeName == volumeName && v.DeviceMountState == operationexecutor.DeviceMountUncertain {
					return true, nil
				}
			}
			return false, nil
		},
	)

	if err != nil {
		t.Fatalf("expected volumes %s to be mounted in uncertain state globally", volumeName)
	}
}

func waitForUncertainPodMount(t *testing.T, volumeName v1.UniqueVolumeName, asw cache.ActualStateOfWorld) {
	// check if volume is locally pod mounted in uncertain state
	err := retryWithExponentialBackOff(
		testOperationBackOffDuration,
		func() (bool, error) {
			allMountedVolumes := asw.GetAllMountedVolumes()
			for _, v := range allMountedVolumes {
				if v.VolumeName == volumeName {
					return true, nil
				}
			}
			return false, nil
		},
	)

	if err != nil {
		t.Fatalf("expected volumes %s to be mounted in uncertain state for pod", volumeName)
	}
}

func waitForMount(
	t *testing.T,
	fakePlugin *volumetesting.FakeVolumePlugin,
	volumeName v1.UniqueVolumeName,
	asw cache.ActualStateOfWorld) {
	err := retryWithExponentialBackOff(
		testOperationBackOffDuration,
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

func waitForVolumeToExistInASW(t *testing.T, volumeName v1.UniqueVolumeName, asw cache.ActualStateOfWorld) {
	err := retryWithExponentialBackOff(
		testOperationBackOffDuration,
		func() (bool, error) {
			if asw.VolumeExists(volumeName) {
				return true, nil
			}
			return false, nil
		},
	)
	if err != nil {
		t.Fatalf("Timed out waiting for volume %q to be exist in asw.", volumeName)
	}
}

func waitForDetach(
	t *testing.T,
	volumeName v1.UniqueVolumeName,
	asw cache.ActualStateOfWorld) {
	err := retryWithExponentialBackOff(
		testOperationBackOffDuration,
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

func createtestClientWithPVPVC(pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim, attachedVolumes ...v1.AttachedVolume) *fake.Clientset {
	fakeClient := &fake.Clientset{}
	if len(attachedVolumes) == 0 {
		attachedVolumes = append(attachedVolumes, v1.AttachedVolume{
			Name:       "fake-plugin/pv",
			DevicePath: "fake/path",
		})
	}
	fakeClient.AddReactor("get", "nodes",
		func(action core.Action) (bool, runtime.Object, error) {
			return true, &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: string(nodeName)},
				Status: v1.NodeStatus{
					VolumesAttached: attachedVolumes,
				},
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
		mount.NewFakeMounter(nil),
		hostutil.NewFakeHostUtil(nil),
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
