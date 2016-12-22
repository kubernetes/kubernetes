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
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

func TestPodVolumesExist(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet

	pods := []*v1.Pod{
		{
			ObjectMeta: v1.ObjectMeta{
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
			ObjectMeta: v1.ObjectMeta{
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
			ObjectMeta: v1.ObjectMeta{
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
	defer func() {
		close(stopCh)
	}()

	kubelet.podManager.SetPods(pods)
	for _, pod := range pods {
		err := kubelet.volumeManager.WaitForAttachAndMount(pod)
		if err != nil {
			t.Errorf("Expected success: %v", err)
		}
	}

	for _, pod := range pods {
		podVolumesExist := kubelet.podVolumesExist(pod.UID)
		if !podVolumesExist {
			t.Errorf(
				"Expected to find volumes for pod %q, but podVolumesExist returned false",
				pod.UID)
		}
	}
}

func TestVolumeAttachAndMountControllerDisabled(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet

	pod := podWithUidNameNsSpec("12345678", "foo", "test", v1.PodSpec{
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
	defer func() {
		close(stopCh)
	}()

	kubelet.podManager.SetPods([]*v1.Pod{pod})
	err := kubelet.volumeManager.WaitForAttachAndMount(pod)
	assert.NoError(t, err)

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		volumehelper.GetUniquePodName(pod))

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
	kubelet := testKubelet.kubelet

	pod := podWithUidNameNsSpec("12345678", "foo", "test", v1.PodSpec{
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
	defer func() {
		close(stopCh)
	}()

	// Add pod
	kubelet.podManager.SetPods([]*v1.Pod{pod})

	// Verify volumes attached
	err := kubelet.volumeManager.WaitForAttachAndMount(pod)
	assert.NoError(t, err)

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		volumehelper.GetUniquePodName(pod))

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
		volumehelper.GetUniquePodName(pod))

	assert.Len(t, podVolumes, 0,
		"Expected volumes to be unmounted and detached. But some volumes are still mounted: %#v", podVolumes)

	assert.NoError(t, volumetest.VerifyTearDownCallCount(
		1 /* expectedTearDownCallCount */, testKubelet.volumePlugin))

	// Verify volumes detached and no longer reported as in use
	assert.NoError(t, waitForVolumeDetach(v1.UniqueVolumeName("fake/vol1"), kubelet.volumeManager))
	assert.True(t, testKubelet.volumePlugin.GetNewAttacherCallCount() >= 1, "Expected plugin NewAttacher to be called at least once")
	assert.NoError(t, volumetest.VerifyDetachCallCount(
		1 /* expectedDetachCallCount */, testKubelet.volumePlugin))
}

func TestVolumeAttachAndMountControllerEnabled(t *testing.T) {
	testKubelet := newTestKubelet(t, true /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.AddReactor("get", "nodes",
		func(action core.Action) (bool, runtime.Object, error) {
			return true, &v1.Node{
				ObjectMeta: v1.ObjectMeta{Name: testKubeletHostname},
				Status: v1.NodeStatus{
					VolumesAttached: []v1.AttachedVolume{
						{
							Name:       "fake/vol1",
							DevicePath: "fake/path",
						},
					}},
				Spec: v1.NodeSpec{ExternalID: testKubeletHostname},
			}, nil
		})
	kubeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("no reaction implemented for %s", action)
	})

	pod := podWithUidNameNsSpec("12345678", "foo", "test", v1.PodSpec{
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
	defer func() {
		close(stopCh)
	}()

	kubelet.podManager.SetPods([]*v1.Pod{pod})

	// Fake node status update
	go simulateVolumeInUseUpdate(
		v1.UniqueVolumeName("fake/vol1"),
		stopCh,
		kubelet.volumeManager)

	assert.NoError(t, kubelet.volumeManager.WaitForAttachAndMount(pod))

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		volumehelper.GetUniquePodName(pod))

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
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.AddReactor("get", "nodes",
		func(action core.Action) (bool, runtime.Object, error) {
			return true, &v1.Node{
				ObjectMeta: v1.ObjectMeta{Name: testKubeletHostname},
				Status: v1.NodeStatus{
					VolumesAttached: []v1.AttachedVolume{
						{
							Name:       "fake/vol1",
							DevicePath: "fake/path",
						},
					}},
				Spec: v1.NodeSpec{ExternalID: testKubeletHostname},
			}, nil
		})
	kubeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("no reaction implemented for %s", action)
	})

	pod := podWithUidNameNsSpec("12345678", "foo", "test", v1.PodSpec{
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
	defer func() {
		close(stopCh)
	}()

	// Add pod
	kubelet.podManager.SetPods([]*v1.Pod{pod})

	// Fake node status update
	go simulateVolumeInUseUpdate(
		v1.UniqueVolumeName("fake/vol1"),
		stopCh,
		kubelet.volumeManager)

	// Verify volumes attached
	assert.NoError(t, kubelet.volumeManager.WaitForAttachAndMount(pod))

	podVolumes := kubelet.volumeManager.GetMountedVolumesForPod(
		volumehelper.GetUniquePodName(pod))

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
		volumehelper.GetUniquePodName(pod))

	assert.Len(t, podVolumes, 0,
		"Expected volumes to be unmounted and detached. But some volumes are still mounted: %#v", podVolumes)

	assert.NoError(t, volumetest.VerifyTearDownCallCount(
		1 /* expectedTearDownCallCount */, testKubelet.volumePlugin))

	// Verify volumes detached and no longer reported as in use
	assert.NoError(t, waitForVolumeDetach(v1.UniqueVolumeName("fake/vol1"), kubelet.volumeManager))
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
