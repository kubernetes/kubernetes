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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
)

func TestListVolumesForPod(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	pod := podWithUIDNameNsSpec("12345678", "foo", "test", v1.PodSpec{
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

	stopCh := runVolumeManager(kubelet)
	defer close(stopCh)

	kubelet.podManager.SetPods([]*v1.Pod{pod})
	err := kubelet.volumeManager.WaitForAttachAndMount(pod)
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
				Volumes: []v1.Volume{
					{
						Name: "vol1",
						VolumeSource: v1.VolumeSource{
							GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
								PDName: "fake-device1",
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
				Volumes: []v1.Volume{
					{
						Name: "vol2",
						VolumeSource: v1.VolumeSource{
							GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
								PDName: "fake-device2",
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
				Volumes: []v1.Volume{
					{
						Name: "vol3",
						VolumeSource: v1.VolumeSource{
							GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
								PDName: "fake-device3",
							},
						},
					},
				},
			},
		},
	}

	stopCh := runVolumeManager(kubelet)
	defer close(stopCh)

	kubelet.podManager.SetPods(pods)
	for _, pod := range pods {
		err := kubelet.volumeManager.WaitForAttachAndMount(pod)
		assert.NoError(t, err)
	}

	for _, pod := range pods {
		podVolumesExist := kubelet.podVolumesExist(pod.UID)
		assert.True(t, podVolumesExist, "pod %q", pod.UID)
	}
}

func TestVolumeAttachAndMountControllerDisabled(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	pod := podWithUIDNameNsSpec("12345678", "foo", "test", v1.PodSpec{
		Volumes: []v1.Volume{
			{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: "fake-device",
					},
				},
			},
		},
	})

	stopCh := runVolumeManager(kubelet)
	defer close(stopCh)

	kubelet.podManager.SetPods([]*v1.Pod{pod})
	err := kubelet.volumeManager.WaitForAttachAndMount(pod)
	assert.NoError(t, err)

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		util.GetUniquePodName(pod))

	expectedPodVolumes := []string{"vol1"}
	assert.Len(t, podVolumes, len(expectedPodVolumes), "Volumes for pod %+v", pod)
	for _, name := range expectedPodVolumes {
		assert.Contains(t, podVolumes, name, "Volumes for pod %+v", pod)
	}
	assert.True(t, testKubelet.volumePlugin.GetNewAttacherCallCount() >= 1, "Expected plugin NewAttacher to be called at least once")
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
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	pod := podWithUIDNameNsSpec("12345678", "foo", "test", v1.PodSpec{
		Volumes: []v1.Volume{
			{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: "fake-device",
					},
				},
			},
		},
	})

	stopCh := runVolumeManager(kubelet)
	defer close(stopCh)

	// Add pod
	kubelet.podManager.SetPods([]*v1.Pod{pod})

	// Verify volumes attached
	err := kubelet.volumeManager.WaitForAttachAndMount(pod)
	assert.NoError(t, err)

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		util.GetUniquePodName(pod))

	expectedPodVolumes := []string{"vol1"}
	assert.Len(t, podVolumes, len(expectedPodVolumes), "Volumes for pod %+v", pod)
	for _, name := range expectedPodVolumes {
		assert.Contains(t, podVolumes, name, "Volumes for pod %+v", pod)
	}

	assert.True(t, testKubelet.volumePlugin.GetNewAttacherCallCount() >= 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyAttachCallCount(
		1 /* expectedAttachCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyMountDeviceCallCount(
		1 /* expectedMountDeviceCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifySetUpCallCount(
		1 /* expectedSetUpCallCount */, testKubelet.volumePlugin))

	// Remove pod
	kubelet.podManager.SetPods([]*v1.Pod{})

	assert.NoError(t, waitForVolumeUnmount(kubelet.volumeManager, pod))

	// Verify volumes unmounted
	podVolumes = kubelet.volumeManager.GetMountedVolumesForPod(
		util.GetUniquePodName(pod))

	assert.Len(t, podVolumes, 0,
		"Expected volumes to be unmounted and detached. But some volumes are still mounted: %#v", podVolumes)

	assert.NoError(t, volumetest.VerifyTearDownCallCount(
		1 /* expectedTearDownCallCount */, testKubelet.volumePlugin))

	// Verify volumes detached and no longer reported as in use
	assert.NoError(t, waitForVolumeDetach(v1.UniqueVolumeName("fake/fake-device"), kubelet.volumeManager))
	assert.True(t, testKubelet.volumePlugin.GetNewAttacherCallCount() >= 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyDetachCallCount(
		1 /* expectedDetachCallCount */, testKubelet.volumePlugin))
}

func TestVolumeAttachAndMountControllerEnabled(t *testing.T) {
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
		Volumes: []v1.Volume{
			{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: "fake-device",
					},
				},
			},
		},
	})

	stopCh := runVolumeManager(kubelet)
	defer close(stopCh)

	kubelet.podManager.SetPods([]*v1.Pod{pod})

	// Fake node status update
	go simulateVolumeInUseUpdate(
		v1.UniqueVolumeName("fake/fake-device"),
		stopCh,
		kubelet.volumeManager)

	assert.NoError(t, kubelet.volumeManager.WaitForAttachAndMount(pod))

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		util.GetUniquePodName(pod))

	expectedPodVolumes := []string{"vol1"}
	assert.Len(t, podVolumes, len(expectedPodVolumes), "Volumes for pod %+v", pod)
	for _, name := range expectedPodVolumes {
		assert.Contains(t, podVolumes, name, "Volumes for pod %+v", pod)
	}
	assert.True(t, testKubelet.volumePlugin.GetNewAttacherCallCount() >= 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyZeroAttachCalls(testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyMountDeviceCallCount(
		1 /* expectedMountDeviceCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifySetUpCallCount(
		1 /* expectedSetUpCallCount */, testKubelet.volumePlugin))
}

func TestVolumeUnmountAndDetachControllerEnabled(t *testing.T) {
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
		Volumes: []v1.Volume{
			{
				Name: "vol1",
				VolumeSource: v1.VolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: "fake-device",
					},
				},
			},
		},
	})

	stopCh := runVolumeManager(kubelet)
	defer close(stopCh)

	// Add pod
	kubelet.podManager.SetPods([]*v1.Pod{pod})

	// Fake node status update
	go simulateVolumeInUseUpdate(
		v1.UniqueVolumeName("fake/fake-device"),
		stopCh,
		kubelet.volumeManager)

	// Verify volumes attached
	assert.NoError(t, kubelet.volumeManager.WaitForAttachAndMount(pod))

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		util.GetUniquePodName(pod))

	expectedPodVolumes := []string{"vol1"}
	assert.Len(t, podVolumes, len(expectedPodVolumes), "Volumes for pod %+v", pod)
	for _, name := range expectedPodVolumes {
		assert.Contains(t, podVolumes, name, "Volumes for pod %+v", pod)
	}

	assert.True(t, testKubelet.volumePlugin.GetNewAttacherCallCount() >= 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyWaitForAttachCallCount(
		1 /* expectedWaitForAttachCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyZeroAttachCalls(testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifyMountDeviceCallCount(
		1 /* expectedMountDeviceCallCount */, testKubelet.volumePlugin))
	assert.NoError(t, volumetest.VerifySetUpCallCount(
		1 /* expectedSetUpCallCount */, testKubelet.volumePlugin))

	// Remove pod
	kubelet.podManager.SetPods([]*v1.Pod{})

	assert.NoError(t, waitForVolumeUnmount(kubelet.volumeManager, pod))

	// Verify volumes unmounted
	podVolumes = kubelet.volumeManager.GetMountedVolumesForPod(
		util.GetUniquePodName(pod))

	assert.Len(t, podVolumes, 0,
		"Expected volumes to be unmounted and detached. But some volumes are still mounted: %#v", podVolumes)

	assert.NoError(t, volumetest.VerifyTearDownCallCount(
		1 /* expectedTearDownCallCount */, testKubelet.volumePlugin))

	// Verify volumes detached and no longer reported as in use
	assert.NoError(t, waitForVolumeDetach(v1.UniqueVolumeName("fake/fake-device"), kubelet.volumeManager))
	assert.True(t, testKubelet.volumePlugin.GetNewAttacherCallCount() >= 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyZeroDetachCallCount(testKubelet.volumePlugin))
}

type stubVolume struct {
	path string
	volume.MetricsNil
}

func (f *stubVolume) GetPath() string {
	return f.path
}

func (f *stubVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{}
}

func (f *stubVolume) CanMount() error {
	return nil
}

func (f *stubVolume) SetUp(fsGroup *int64) error {
	return nil
}

func (f *stubVolume) SetUpAt(dir string, fsGroup *int64) error {
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

func (f stubBlockVolume) MapDevice(devicePath, globalMapPath, volumeMapPath, volumeMapName string, podUID types.UID) error {
	return nil
}

func (f *stubBlockVolume) TearDownDevice(mapPath string, devicePath string) error {
	return nil
}
