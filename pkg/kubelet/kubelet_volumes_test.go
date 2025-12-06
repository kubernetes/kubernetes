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

package kubelet

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	utiltesting "k8s.io/client-go/util/testing"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/populator"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/test/utils/ktesting"

	"k8s.io/klog/v2"
)

func TestListVolumesForPod(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	pod := podWithUIDNameNsSpec("12345678", "foo", "test", v1.PodSpec{
		Containers: []v1.Container{
			{
				Name: "container1",
				VolumeMounts: []v1.VolumeMount{
					{
						Name:      "vol1",
						MountPath: "/mnt/vol1",
					},
					{
						Name:      "vol2",
						MountPath: "/mnt/vol2",
					},
				},
			},
		},
		Volumes: []v1.Volume{
			{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: "fake-device1",
					},
				},
			},
			{
				Name: "vol2",
				VolumeSource: v1.VolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: "fake-device2",
					},
				},
			},
		},
	})

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	go kubelet.volumeManager.Run(tCtx, kubelet.sourcesReady)

	kubelet.podManager.SetPods([]*v1.Pod{pod})
	err := kubelet.volumeManager.WaitForAttachAndMount(context.Background(), pod)
	assert.NoError(t, err)

	podName := util.GetUniquePodName(pod)

	volumesToReturn, volumeExsit := kubelet.ListVolumesForPod(types.UID(podName))
	assert.True(t, volumeExsit, "expected to find volumes for pod %q", podName)

	outerVolumeSpecName1 := "vol1"
	assert.NotNil(t, volumesToReturn[outerVolumeSpecName1], "key %s", outerVolumeSpecName1)

	outerVolumeSpecName2 := "vol2"
	assert.NotNil(t, volumesToReturn[outerVolumeSpecName2], "key %s", outerVolumeSpecName2)
}

func TestPodVolumesExist(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	pods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod1",
				UID:  "pod1uid",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "container1",
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "vol1",
								MountPath: "/mnt/vol1",
							},
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "vol1",
						VolumeSource: v1.VolumeSource{
							RBD: &v1.RBDVolumeSource{
								RBDImage: "fake1",
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod2",
				UID:  "pod2uid",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "container2",
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "vol2",
								MountPath: "/mnt/vol2",
							},
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "vol2",
						VolumeSource: v1.VolumeSource{
							RBD: &v1.RBDVolumeSource{
								RBDImage: "fake2",
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod3",
				UID:  "pod3uid",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "container3",
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "vol3",
								MountPath: "/mnt/vol3",
							},
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "vol3",
						VolumeSource: v1.VolumeSource{
							RBD: &v1.RBDVolumeSource{
								RBDImage: "fake3",
							},
						},
					},
				},
			},
		},
	}

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	go kubelet.volumeManager.Run(tCtx, kubelet.sourcesReady)

	kubelet.podManager.SetPods(pods)
	for _, pod := range pods {
		err := kubelet.volumeManager.WaitForAttachAndMount(context.Background(), pod)
		assert.NoError(t, err)
	}

	for _, pod := range pods {
		podVolumesExist := kubelet.podVolumesExist(pod.UID)
		assert.True(t, podVolumesExist, "pod %q", pod.UID)
	}
}

func TestPodVolumeDeadlineAttachAndMount(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	testKubelet := newTestKubeletWithImageList(t, nil /*imageList*/, false, /* controllerAttachDetachEnabled */
		false /*initFakeVolumePlugin*/, true /*localStorageCapacityIsolation*/, false /*excludePodAdmitHandlers*/, false /*enableResizing*/)

	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	// any test cases added here should have volumes that fail to mount
	pods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod1",
				UID:  "pod1uid",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "container1",
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "vol1",
								MountPath: "/mnt/vol1",
							},
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "vol1",
						VolumeSource: v1.VolumeSource{
							Secret: &v1.SecretVolumeSource{
								SecretName: "non-existent",
							},
						},
					},
				},
			},
		},
	}

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	go kubelet.volumeManager.Run(tCtx, kubelet.sourcesReady)

	kubelet.podManager.SetPods(pods)
	for _, pod := range pods {
		start := time.Now()
		// ensure our context times out quickly
		ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(2*time.Second))
		err := kubelet.volumeManager.WaitForAttachAndMount(ctx, pod)
		delta := time.Since(start)
		// the standard timeout is 2 minutes, so if it's just a few seconds we know that the context timeout was the cause
		assert.Lessf(t, delta, 10*time.Second, "WaitForAttachAndMount should timeout when the context is cancelled")
		assert.ErrorIs(t, err, context.DeadlineExceeded)
		cancel()
	}
}

func TestPodVolumeDeadlineUnmount(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	testKubelet := newTestKubeletWithImageList(t, nil /*imageList*/, false, /* controllerAttachDetachEnabled */
		true /*initFakeVolumePlugin*/, true /*localStorageCapacityIsolation*/, false /*excludePodAdmitHandlers*/, false /*enableResizing*/)

	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	// any test cases added here should have volumes that succeed at mounting
	pods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod1",
				UID:  "pod1uid",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "container1",
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "vol1",
								MountPath: "/mnt/vol1",
							},
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "vol1",
						VolumeSource: v1.VolumeSource{
							RBD: &v1.RBDVolumeSource{
								RBDImage: "fake-device",
							},
						},
					},
				},
			},
		},
	}

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	go kubelet.volumeManager.Run(tCtx, kubelet.sourcesReady)

	kubelet.podManager.SetPods(pods)
	for i, pod := range pods {
		if err := kubelet.volumeManager.WaitForAttachAndMount(context.Background(), pod); err != nil {
			t.Fatalf("pod %d failed: %v", i, err)
		}
		start := time.Now()
		// ensure our context times out quickly
		ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(2*time.Second))
		err := kubelet.volumeManager.WaitForUnmount(ctx, pod)
		delta := time.Since(start)
		// the standard timeout is 2 minutes, so if it's just a few seconds we know that the context timeout was the cause
		assert.Lessf(t, delta, 10*time.Second, "WaitForUnmount should timeout when the context is cancelled")
		assert.ErrorIs(t, err, context.DeadlineExceeded)
		cancel()
	}
}

func TestVolumeAttachAndMountControllerDisabled(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	pod := podWithUIDNameNsSpec("12345678", "foo", "test", v1.PodSpec{
		Containers: []v1.Container{
			{
				Name: "container1",
				VolumeMounts: []v1.VolumeMount{
					{
						Name:      "vol1",
						MountPath: "/mnt/vol1",
					},
				},
			},
		},
		Volumes: []v1.Volume{
			{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					RBD: &v1.RBDVolumeSource{
						RBDImage: "fake",
					},
				},
			},
		},
	})

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	go kubelet.volumeManager.Run(tCtx, kubelet.sourcesReady)

	kubelet.podManager.SetPods([]*v1.Pod{pod})
	err := kubelet.volumeManager.WaitForAttachAndMount(context.Background(), pod)
	assert.NoError(t, err)

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		util.GetUniquePodName(pod))

	expectedPodVolumes := []string{"vol1"}
	assert.Len(t, podVolumes, len(expectedPodVolumes), "Volumes for pod %+v", pod)
	for _, name := range expectedPodVolumes {
		assert.Contains(t, podVolumes, name, "Volumes for pod %+v", pod)
	}
	assert.GreaterOrEqual(t, testKubelet.volumePlugin.GetNewAttacherCallCount(), 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyAttachCallCount(
		1 /* expectedAttachCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyMountDeviceCallCount(
		1 /* expectedMountDeviceCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifySetUpCallCount(
		1 /* expectedSetUpCallCount */, testKubelet.volumePlugin))
}

func TestVolumeUnmountAndDetachControllerDisabled(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	pod := podWithUIDNameNsSpec("12345678", "foo", "test", v1.PodSpec{
		Containers: []v1.Container{
			{
				Name: "container1",
				VolumeMounts: []v1.VolumeMount{
					{
						Name:      "vol1",
						MountPath: "/mnt/vol1",
					},
				},
			},
		},
		Volumes: []v1.Volume{
			{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					RBD: &v1.RBDVolumeSource{
						RBDImage: "fake-device",
					},
				},
			},
		},
	})

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	go kubelet.volumeManager.Run(tCtx, kubelet.sourcesReady)

	// Add pod
	kubelet.podManager.SetPods([]*v1.Pod{pod})

	// Verify volumes attached
	err := kubelet.volumeManager.WaitForAttachAndMount(context.Background(), pod)
	assert.NoError(t, err)

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		util.GetUniquePodName(pod))

	expectedPodVolumes := []string{"vol1"}
	assert.Len(t, podVolumes, len(expectedPodVolumes), "Volumes for pod %+v", pod)
	for _, name := range expectedPodVolumes {
		assert.Contains(t, podVolumes, name, "Volumes for pod %+v", pod)
	}

	assert.GreaterOrEqual(t, testKubelet.volumePlugin.GetNewAttacherCallCount(), 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyAttachCallCount(
		1 /* expectedAttachCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyMountDeviceCallCount(
		1 /* expectedMountDeviceCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifySetUpCallCount(
		1 /* expectedSetUpCallCount */, testKubelet.volumePlugin))

	// Remove pod
	// TODO: technically waitForVolumeUnmount
	kubelet.podWorkers.(*fakePodWorkers).setPodRuntimeBeRemoved(pod.UID)
	kubelet.podManager.SetPods([]*v1.Pod{})

	assert.NoError(t, kubelet.volumeManager.WaitForUnmount(context.Background(), pod))

	// Verify volumes unmounted
	hasMountedVolumes := kubelet.volumeManager.HasPossiblyMountedVolumesForPod(
		util.GetUniquePodName(pod))

	assert.False(t, hasMountedVolumes, "Expected volumes to be unmounted and detached. But some volumes are still mounted")

	assert.NoError(t, volumetest.VerifyTearDownCallCount(
		1 /* expectedTearDownCallCount */, testKubelet.volumePlugin))

	// Verify volumes detached and no longer reported as in use
	assert.NoError(t, waitForVolumeDetach(v1.UniqueVolumeName("fake/fake-device"), kubelet.volumeManager))
	assert.GreaterOrEqual(t, testKubelet.volumePlugin.GetNewAttacherCallCount(), 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyDetachCallCount(
		1 /* expectedDetachCallCount */, testKubelet.volumePlugin))
}

func TestVolumeAttachAndMountControllerEnabled(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	testKubelet := newTestKubelet(t, true /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.AddReactor("get", "nodes",
		func(action core.Action) (bool, runtime.Object, error) {
			return true, &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Status: v1.NodeStatus{
					VolumesAttached: []v1.AttachedVolume{
						{
							Name:       "fake/fake-device",
							DevicePath: "fake/path",
						},
					}},
			}, nil
		})
	kubeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("no reaction implemented for %s", action)
	})

	pod := podWithUIDNameNsSpec("12345678", "foo", "test", v1.PodSpec{
		Containers: []v1.Container{
			{
				Name: "container1",
				VolumeMounts: []v1.VolumeMount{
					{
						Name:      "vol1",
						MountPath: "/mnt/vol1",
					},
				},
			},
		},
		Volumes: []v1.Volume{
			{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					RBD: &v1.RBDVolumeSource{
						RBDImage: "fake-device",
					},
				},
			},
		},
	})

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	go kubelet.volumeManager.Run(tCtx, kubelet.sourcesReady)

	kubelet.podManager.SetPods([]*v1.Pod{pod})

	// Fake node status update
	go simulateVolumeInUseUpdate(
		v1.UniqueVolumeName("fake/fake-device"),
		tCtx.Done(),
		kubelet.volumeManager)

	assert.NoError(t, kubelet.volumeManager.WaitForAttachAndMount(context.Background(), pod))

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		util.GetUniquePodName(pod))
	hasVolumes := kubelet.volumeManager.HasPossiblyMountedVolumesForPod(
		util.GetUniquePodName(pod))
	assert.True(t, hasVolumes, "HasPossiblyMountedVolumesForPod should return true")

	expectedPodVolumes := []string{"vol1"}
	assert.Len(t, podVolumes, len(expectedPodVolumes), "Volumes for pod %+v", pod)
	for _, name := range expectedPodVolumes {
		assert.Contains(t, podVolumes, name, "Volumes for pod %+v", pod)
	}
	assert.GreaterOrEqual(t, testKubelet.volumePlugin.GetNewAttacherCallCount(), 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyZeroAttachCalls(testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyMountDeviceCallCount(
		1 /* expectedMountDeviceCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifySetUpCallCount(
		1 /* expectedSetUpCallCount */, testKubelet.volumePlugin))
}

func TestVolumeUnmountAndDetachControllerEnabled(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	testKubelet := newTestKubelet(t, true /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.AddReactor("get", "nodes",
		func(action core.Action) (bool, runtime.Object, error) {
			return true, &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: testKubeletHostname},
				Status: v1.NodeStatus{
					VolumesAttached: []v1.AttachedVolume{
						{
							Name:       "fake/fake-device",
							DevicePath: "fake/path",
						},
					}},
			}, nil
		})
	kubeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("no reaction implemented for %s", action)
	})

	pod := podWithUIDNameNsSpec("12345678", "foo", "test", v1.PodSpec{
		Containers: []v1.Container{
			{
				Name: "container1",
				VolumeMounts: []v1.VolumeMount{
					{
						Name:      "vol1",
						MountPath: "/mnt/vol1",
					},
				},
			},
		},
		Volumes: []v1.Volume{
			{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					RBD: &v1.RBDVolumeSource{
						RBDImage: "fake-device",
					},
				},
			},
		},
	})

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	go kubelet.volumeManager.Run(tCtx, kubelet.sourcesReady)

	// Add pod
	kubelet.podManager.SetPods([]*v1.Pod{pod})

	// Fake node status update
	go simulateVolumeInUseUpdate(
		v1.UniqueVolumeName("fake/fake-device"),
		tCtx.Done(),
		kubelet.volumeManager)

	// Verify volumes attached
	assert.NoError(t, kubelet.volumeManager.WaitForAttachAndMount(context.Background(), pod))

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		util.GetUniquePodName(pod))
	hasVolumes := kubelet.volumeManager.HasPossiblyMountedVolumesForPod(
		util.GetUniquePodName(pod))
	assert.True(t, hasVolumes, "HasPossiblyMountedVolumesForPod should return true")

	expectedPodVolumes := []string{"vol1"}
	assert.Len(t, podVolumes, len(expectedPodVolumes), "Volumes for pod %+v", pod)
	for _, name := range expectedPodVolumes {
		assert.Contains(t, podVolumes, name, "Volumes for pod %+v", pod)
	}

	assert.GreaterOrEqual(t, testKubelet.volumePlugin.GetNewAttacherCallCount(), 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyZeroAttachCalls(testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyMountDeviceCallCount(
		1 /* expectedMountDeviceCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifySetUpCallCount(
		1 /* expectedSetUpCallCount */, testKubelet.volumePlugin))

	// Remove pod
	kubelet.podWorkers.(*fakePodWorkers).setPodRuntimeBeRemoved(pod.UID)
	kubelet.podManager.SetPods([]*v1.Pod{})

	assert.NoError(t, waitForVolumeUnmount(kubelet.volumeManager, pod))

	// Verify volumes unmounted
	podVolumes = kubelet.volumeManager.GetMountedVolumesForPod(
		util.GetUniquePodName(pod))
	hasVolumes = kubelet.volumeManager.HasPossiblyMountedVolumesForPod(
		util.GetUniquePodName(pod))
	assert.False(t, hasVolumes, "HasPossiblyMountedVolumesForPod should return false")

	assert.Empty(t, podVolumes,
		"Expected volumes to be unmounted and detached. But some volumes are still mounted")

	assert.NoError(t, volumetest.VerifyTearDownCallCount(
		1 /* expectedTearDownCallCount */, testKubelet.volumePlugin))

	// Verify volumes detached and no longer reported as in use
	assert.NoError(t, waitForVolumeDetach(v1.UniqueVolumeName("fake/fake-device"), kubelet.volumeManager))
	assert.GreaterOrEqual(t, testKubelet.volumePlugin.GetNewAttacherCallCount(), 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyZeroDetachCallCount(testKubelet.volumePlugin))
}

type stubVolume struct {
	path       string
	attributes volume.Attributes
	volume.MetricsNil
}

func (f *stubVolume) GetPath() string {
	return f.path
}

func (f *stubVolume) GetAttributes() volume.Attributes {
	return f.attributes
}

func (f *stubVolume) SetUp(mounterArgs volume.MounterArgs) error {
	return nil
}

func (f *stubVolume) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	return nil
}

type stubBlockVolume struct {
	dirPath string
	volName string
}

func (f *stubBlockVolume) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	return "", nil
}

func (f *stubBlockVolume) GetPodDeviceMapPath() (string, string) {
	return f.dirPath, f.volName
}

func (f *stubBlockVolume) SetUpDevice() (string, error) {
	return "", nil
}

func (f stubBlockVolume) MapPodDevice() error {
	return nil
}

func (f *stubBlockVolume) TearDownDevice(mapPath string, devicePath string) error {
	return nil
}

func (f *stubBlockVolume) UnmapPodDevice() error {
	return nil
}

func (f *stubBlockVolume) SupportsMetrics() bool {
	return false
}

func (f *stubBlockVolume) GetMetrics() (*volume.Metrics, error) {
	return nil, nil
}

func TestStaticPodVolumeProcessShouldNotInterleave(t *testing.T) {
	// this test exercises the following scenario:
	// we have two static pods with uid 1 and 2, both pods have the same
	// name, both pods are identical in spec, and they have a volume
	// we will refer to these pods as pod-1, and pod-2 respectively
	// a) pod-1 starts
	// b) populator for the desired state of the world comes across pod-1
	//    and starts processing its volume by calling AddPodToVolume
	// c) complete the sync for pod-1 so it is in terminating state, and we
	//    let it remain in this state so the test can exercise its steps
	// d) start pod-2, since pod-1 with the same name is in terminating
	//    state, pod-2 will not be deemed ready to start yet by pod worker
	// e) as the populator for the desired state of the world runs
	//    asynchronously, it comes across pod-2, but it should not start
	//    processing volumes of pod-2 by calling AddPodToVolume just yet
	// f) drain all workers for pod-1, and pod-1 transitions to terminated state
	// g) wait for the populator to start processing the volumes
	//    of pod-2 by calling AddPodToVolume
	clientset := &fake.Clientset{}
	podManager := kubepod.NewBasicPodManager()

	// so we can force pod-1 to remain in terminating state
	timedPodWorkers, _ := createTimeIncrementingPodWorkers()
	podWorkers := timedPodWorkers.w
	intreeToCSITranslator := csitrans.New()
	csiMigratedPluginManager := csimigration.NewPluginManager(intreeToCSITranslator, utilfeature.DefaultFeatureGate)
	seLinuxTranslator := util.NewSELinuxLabelTranslator()

	// we need a fake plugin and an initialized VolumePluginMgr
	tmpDir, err := utiltesting.MkTmpdir("volumeManagerTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			t.Fatalf("failed to remove temp dir: %v", err)
		}
	}()
	// TODO (#51147) inject mock prober
	fakeVolumeHost := volumetest.NewFakeKubeletVolumeHost(t, tmpDir, clientset, nil)
	fakeVolumeHost.WithNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "fake"}})
	fakePlugin := &volumetest.FakeVolumePlugin{
		PluginName: "fake",
		Host:       nil,
		CanSupportFn: func(spec *volume.Spec) bool {
			return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.RBD != nil) ||
				(spec.Volume != nil && spec.Volume.RBD != nil)
		},
	}
	volumePluginMgr := &volume.VolumePluginMgr{}
	if err := volumePluginMgr.InitPlugins([]volume.VolumePlugin{fakePlugin}, nil /* prober */, fakeVolumeHost); err != nil {
		t.Fatalf("volume plugin manager failed to initialize")
	}

	// this wraps the AddPodToVolume method of the DesiredStateOfWorld in
	// use by the populator so we can record how it's being invoked
	desired := &trackingAddPodToVolume{
		DesiredStateOfWorld: cache.NewDesiredStateOfWorld(volumePluginMgr, seLinuxTranslator),
	}
	actual := cache.NewActualStateOfWorld("fake", volumePluginMgr)

	dswPopulator := populator.NewDesiredStateOfWorldPopulator(
		clientset,
		10*time.Millisecond, // the populator will poll every 10ms
		podManager,          // shared PodManager
		podWorkers,          // shared PodStateProvider
		desired,             // shared cache.DesiredStateOfWorld in use by the populator
		actual,
		csiMigratedPluginManager,
		intreeToCSITranslator,
		volumePluginMgr,
	)

	// returns a static pod with volume(s)
	newPodFn := func(uid types.UID, name string) *v1.Pod {
		pod := newStaticPod(string(uid), name)
		pod.Spec.Containers = []v1.Container{
			{
				Name: "container1",
				VolumeMounts: []v1.VolumeMount{
					{
						Name:      "vol1",
						MountPath: "/mnt/vol1",
					},
				},
			},
		}
		pod.Spec.Volumes = []v1.Volume{
			{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					RBD: &v1.RBDVolumeSource{
						RBDImage: "fake1",
					},
				},
			},
		}
		return pod
	}

	// waits up to timeout to observe if AddPodToVolume has been invoked
	waitForAddPodToVolumeInvoked := func(timeout time.Duration) error {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		return wait.PollUntilContextCancel(ctx, 5*time.Millisecond, true, func(ctx context.Context) (bool, error) {
			var done bool
			desired.visit(func(r recorded) {
				if len(r.added) > 0 {
					done = true
				}
			})
			return done, nil
		})
	}

	// run the populator in its own goroutine
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	populatorDone := make(chan struct{})
	go func() {
		defer close(populatorDone)
		alwaysReady := config.NewSourcesReady(func(_ sets.Set[string]) bool { return true })
		dswPopulator.Run(ctx, alwaysReady)
	}()

	// we have two static pods having the same name
	name := "static-pod-foo"
	uid1, uid2 := types.UID("1"), types.UID("2")
	pod1, pod2 := newPodFn(uid1, name), newPodFn(uid2, name)
	fullname := kubecontainer.BuildPodFullName(pod1.Name, pod1.Namespace)

	if !kubetypes.IsStaticPod(pod1) || !kubetypes.IsStaticPod(pod2) {
		t.Fatalf("wrong test setup - must be static pods")
	}

	// a) start the first static pod, pod-1
	// manually add pod-1 to pod manager cache
	podManager.AddPod(pod1)
	timedPodWorkers.UpdatePod(UpdatePodOptions{
		Pod:        pod1,
		UpdateType: kubetypes.SyncPodUpdate,
	})
	podWorkers.podLock.Lock()
	// we should observe the static pod running
	if status := podWorkers.podSyncStatuses[uid1]; status == nil || status.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", status)
	}
	if uidGot, ok := podWorkers.startedStaticPodsByFullname[fullname]; !ok || uidGot != uid1 {
		t.Fatalf("expected pod-1 to be in startedStaticPodsByFullname")
	}
	if waiting := podWorkers.waitingToStartStaticPodsByFullname[fullname]; len(waiting) > 0 {
		t.Fatalf("did not expect pod-1 to be in waitingToStartStaticPodsByFullname")
	}
	podWorkers.podLock.Unlock()

	// b) we expect the volume(s) of pod-1 to be processed by the populator
	// wait for 3 second(s) at most, to minimize flakes
	if err := waitForAddPodToVolumeInvoked(3 * time.Second); err != nil {
		t.Fatalf("timed out while waiting for AddPodToVolume to be invoked: %v", err)
	}
	desired.visit(func(r recorded) {
		if len(r.added) != 1 || r.added[0] != uid1 || len(r.error) > 0 {
			t.Errorf("expected volumes of pod-1 to be processed - %#v", r)
		}
	})
	desired.reset()

	// c) transition pod-1 to terminating, and let it stay in terminating state
	timedPodWorkers.PauseWorkers(uid1)
	podWorkers.completeSync(uid1)
	podWorkers.podLock.Lock()
	if status := podWorkers.podSyncStatuses[uid1]; status == nil || !status.IsTerminationStarted() || !status.IsTerminationRequested() {
		t.Fatalf("expected pod-1 to be in terminating state: %#v", status)
	}
	// pod-1 should still be in startedStaticPodsByFullname
	if uidGot, ok := podWorkers.startedStaticPodsByFullname[fullname]; !ok || uidGot != uid1 {
		t.Fatalf("expected pod-1 to be in startedStaticPodsByFullname")
	}
	podWorkers.podLock.Unlock()

	// d) start static pod pod-2
	podManager.AddPod(pod2)
	timedPodWorkers.UpdatePod(UpdatePodOptions{
		Pod:        pod2,
		UpdateType: kubetypes.SyncPodUpdate,
	})
	podWorkers.podLock.Lock()
	if status := podWorkers.podSyncStatuses[uid2]; status == nil || status.IsTerminated() {
		t.Fatalf("unexpected pod state: %#v", status)
	}
	if uidGot, ok := podWorkers.startedStaticPodsByFullname[fullname]; ok && uidGot == uid2 {
		t.Fatalf("did not expect pod-2 in startedStaticPodsByFullname yet")
	}
	if waiting := podWorkers.waitingToStartStaticPodsByFullname[fullname]; len(waiting) != 1 || waiting[0] != uid2 {
		t.Fatalf("expected pod-2 in waitingToStartStaticPodsByFullname")
	}
	podWorkers.podLock.Unlock()

	// e) wait for at most 3s to see if AddPodToVolume was invoked
	// we don't expect AddPodToVolume to be called by the populator
	// for pod-2 volumes just yet
	if err := waitForAddPodToVolumeInvoked(3 * time.Second); err == nil {
		t.Errorf("did not expect AddPodToVolume to be invoked")
	}
	desired.visit(func(r recorded) {
		if len(r.added) > 0 || len(r.error) > 0 {
			t.Fatalf("did not expect pod-2 volume(s) to be processed while pod-1 is terminating - %#v", r)
		}
	})
	desired.reset()

	// f) transition pod-1 from terminating to terminated
	timedPodWorkers.ReleaseWorkers(uid1)
	drainAllWorkers(podWorkers)
	podWorkers.podLock.Lock()
	if status := podWorkers.podSyncStatuses[uid1]; status == nil || !status.IsTerminated() {
		t.Fatalf("unexpected podpod-1 to be in terminated state: %#v", status)
	}
	podWorkers.podLock.Unlock()

	// g) now that pod-1 is in terminated state, we expect the populator to
	// start processing volume(s) of pod-2 by calling AddPodToVolume.
	// wait for at most 3s, to minimize flakes
	if err := waitForAddPodToVolumeInvoked(3 * time.Second); err != nil {
		t.Errorf("timed out while waiting for AddPodToVolume to be invoked: %v", err)
	}
	desired.visit(func(r recorded) {
		if len(r.added) != 1 || r.added[0] != uid2 || len(r.error) > 0 {
			t.Errorf("expected pod-2 volume(s) to be processed by the populator - %#v", r)
		}
	})

	// wait for the populator to return
	cancel()
	select {
	case <-populatorDone:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("timed out waiting for the populator to be done")
	}
}

type recorded struct {
	// when AddPodToVolume is invoked, the uid of the given pod is appended
	added []types.UID
	// if AddErrorToPod is invoked, the error is saved here
	error string
}
type trackingAddPodToVolume struct {
	cache.DesiredStateOfWorld

	// recorded values
	lock  sync.Mutex
	count recorded
}

func (dsw *trackingAddPodToVolume) AddPodToVolume(logger klog.Logger, podName volumetypes.UniquePodName, pod *v1.Pod,
	volumeSpec *volume.Spec, outerVolumeSpecName string, volumeGIDValue string,
	seLinuxContainerContexts []*v1.SELinuxOptions) (v1.UniqueVolumeName, error) {
	dsw.lock.Lock()
	dsw.count.added = append(dsw.count.added, pod.UID)
	dsw.lock.Unlock()

	return dsw.DesiredStateOfWorld.AddPodToVolume(logger, podName, pod, volumeSpec, outerVolumeSpecName, volumeGIDValue, seLinuxContainerContexts)
}

func (dsw *trackingAddPodToVolume) AddErrorToPod(podName volumetypes.UniquePodName, err string) {
	dsw.lock.Lock()
	if len(dsw.count.error) == 0 {
		dsw.count.error = err
	}
	dsw.lock.Unlock()

	dsw.DesiredStateOfWorld.AddErrorToPod(podName, err)
}

// the following methods are intended to be called by the test
func (dsw *trackingAddPodToVolume) visit(f func(r recorded)) {
	dsw.lock.Lock()
	defer dsw.lock.Unlock()
	f(dsw.count)
}

func (dsw *trackingAddPodToVolume) reset() {
	dsw.lock.Lock()
	defer dsw.lock.Unlock()
	dsw.count = recorded{}
}
