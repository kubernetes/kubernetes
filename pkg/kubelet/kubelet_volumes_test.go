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
package kubelet

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestMountExternalVolumes(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	plug := &volumetest.FakeVolumePlugin{PluginName: "fake", Host: nil}
	kubelet.volumePluginMgr.InitPlugins([]volume.VolumePlugin{plug}, &volumeHost{kubelet})

	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "test",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name:         "vol1",
					VolumeSource: api.VolumeSource{},
				},
			},
		},
	}
	podVolumes, err := kubelet.mountExternalVolumes(&pod)
	if err != nil {
		t.Errorf("Expected success: %v", err)
	}
	expectedPodVolumes := []string{"vol1"}
	if len(expectedPodVolumes) != len(podVolumes) {
		t.Errorf("Unexpected volumes. Expected %#v got %#v.  Manifest was: %#v", expectedPodVolumes, podVolumes, pod)
	}
	for _, name := range expectedPodVolumes {
		if _, ok := podVolumes[name]; !ok {
			t.Errorf("api.Pod volumes map is missing key: %s. %#v", name, podVolumes)
		}
	}
	if plug.NewAttacherCallCount != 1 {
		t.Errorf("Expected plugin NewAttacher to be called %d times but got %d", 1, plug.NewAttacherCallCount)
	}
}

func TestGetPodVolumesFromDisk(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	plug := &volumetest.FakeVolumePlugin{PluginName: "fake", Host: nil}
	kubelet.volumePluginMgr.InitPlugins([]volume.VolumePlugin{plug}, &volumeHost{kubelet})

	volsOnDisk := []struct {
		podUID  types.UID
		volName string
	}{
		{"pod1", "vol1"},
		{"pod1", "vol2"},
		{"pod2", "vol1"},
	}

	expectedPaths := []string{}
	for i := range volsOnDisk {
		fv := volumetest.FakeVolume{PodUID: volsOnDisk[i].podUID, VolName: volsOnDisk[i].volName, Plugin: plug}
		fv.SetUp(nil)
		expectedPaths = append(expectedPaths, fv.GetPath())
	}

	volumesFound := kubelet.getPodVolumesFromDisk()
	if len(volumesFound) != len(expectedPaths) {
		t.Errorf("Expected to find %d unmounters, got %d", len(expectedPaths), len(volumesFound))
	}
	for _, ep := range expectedPaths {
		found := false
		for _, cl := range volumesFound {
			if ep == cl.Unmounter.GetPath() {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Could not find a volume with path %s", ep)
		}
	}
	if plug.NewDetacherCallCount != len(volsOnDisk) {
		t.Errorf("Expected plugin NewDetacher to be called %d times but got %d", len(volsOnDisk), plug.NewDetacherCallCount)
	}
}

// Test for https://github.com/kubernetes/kubernetes/pull/19600
func TestCleanupOrphanedVolumes(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubelet.mounter = &mount.FakeMounter{}
	kubeClient := testKubelet.fakeKubeClient
	plug := &volumetest.FakeVolumePlugin{PluginName: "fake", Host: nil}
	kubelet.volumePluginMgr.InitPlugins([]volume.VolumePlugin{plug}, &volumeHost{kubelet})

	// create a volume "on disk"
	volsOnDisk := []struct {
		podUID  types.UID
		volName string
	}{
		{"podUID", "myrealvol"},
	}

	pathsOnDisk := []string{}
	for i := range volsOnDisk {
		fv := volumetest.FakeVolume{PodUID: volsOnDisk[i].podUID, VolName: volsOnDisk[i].volName, Plugin: plug}
		fv.SetUp(nil)
		pathsOnDisk = append(pathsOnDisk, fv.GetPath())
	}

	// store the claim in fake kubelet database
	claim := api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "myclaim",
			Namespace: "test",
		},
		Spec: api.PersistentVolumeClaimSpec{
			VolumeName: "myrealvol",
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimBound,
		},
	}
	kubeClient.ReactionChain = fake.NewSimpleClientset(&api.PersistentVolumeClaimList{Items: []api.PersistentVolumeClaim{
		claim,
	}}).ReactionChain

	// Create a pod referencing the volume via a PersistentVolumeClaim
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "podUID",
			Name:      "pod",
			Namespace: "test",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "myvolumeclaim",
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: "myclaim",
						},
					},
				},
			},
		},
	}

	// The pod is pending and not running yet. Test that cleanupOrphanedVolumes
	// won't remove the volume from disk if the volume is referenced only
	// indirectly by a claim.
	err := kubelet.cleanupOrphanedVolumes([]*api.Pod{&pod}, []*kubecontainer.Pod{})
	if err != nil {
		t.Errorf("cleanupOrphanedVolumes failed: %v", err)
	}

	volumesFound := kubelet.getPodVolumesFromDisk()
	if len(volumesFound) != len(pathsOnDisk) {
		t.Errorf("Expected to find %d unmounters, got %d", len(pathsOnDisk), len(volumesFound))
	}
	for _, ep := range pathsOnDisk {
		found := false
		for _, cl := range volumesFound {
			if ep == cl.Unmounter.GetPath() {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Could not find a volume with path %s", ep)
		}
	}

	// The pod is deleted -> kubelet should delete the volume
	err = kubelet.cleanupOrphanedVolumes([]*api.Pod{}, []*kubecontainer.Pod{})
	if err != nil {
		t.Errorf("cleanupOrphanedVolumes failed: %v", err)
	}
	volumesFound = kubelet.getPodVolumesFromDisk()
	if len(volumesFound) != 0 {
		t.Errorf("Expected to find 0 unmounters, got %d", len(volumesFound))
	}
	for _, cl := range volumesFound {
		t.Errorf("Found unexpected volume %s", cl.Unmounter.GetPath())
	}
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

func (f *stubVolume) SetUp(fsGroup *int64) error {
	return nil
}

func (f *stubVolume) SetUpAt(dir string, fsGroup *int64) error {
	return nil
}

func TestMakeVolumeMounts(t *testing.T) {
	container := api.Container{
		VolumeMounts: []api.VolumeMount{
			{
				MountPath: "/etc/hosts",
				Name:      "disk",
				ReadOnly:  false,
			},
			{
				MountPath: "/mnt/path3",
				Name:      "disk",
				ReadOnly:  true,
			},
			{
				MountPath: "/mnt/path4",
				Name:      "disk4",
				ReadOnly:  false,
			},
			{
				MountPath: "/mnt/path5",
				Name:      "disk5",
				ReadOnly:  false,
			},
		},
	}

	podVolumes := kubecontainer.VolumeMap{
		"disk":  kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
		"disk4": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/host"}},
		"disk5": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/var/lib/kubelet/podID/volumes/empty/disk5"}},
	}

	pod := api.Pod{
		Spec: api.PodSpec{
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
		},
	}

	mounts, _ := makeMounts(&pod, "/pod", &container, "fakepodname", "", "", podVolumes)

	expectedMounts := []kubecontainer.Mount{
		{
			"disk",
			"/etc/hosts",
			"/mnt/disk",
			false,
			false,
		},
		{
			"disk",
			"/mnt/path3",
			"/mnt/disk",
			true,
			false,
		},
		{
			"disk4",
			"/mnt/path4",
			"/mnt/host",
			false,
			false,
		},
		{
			"disk5",
			"/mnt/path5",
			"/var/lib/kubelet/podID/volumes/empty/disk5",
			false,
			false,
		},
	}
	if !reflect.DeepEqual(mounts, expectedMounts) {
		t.Errorf("Unexpected mounts: Expected %#v got %#v.  Container was: %#v", expectedMounts, mounts, container)
	}
}
