/*
Copyright 2017 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume/util/subpath"
)

func TestMakeMountsWindows(t *testing.T) {
	container := v1.Container{
		VolumeMounts: []v1.VolumeMount{
			{
				MountPath: "c:/etc/hosts",
				Name:      "disk",
				ReadOnly:  false,
			},
			{
				MountPath: "c:/mnt/path3",
				Name:      "disk",
				ReadOnly:  true,
			},
			{
				MountPath: "c:/mnt/path4",
				Name:      "disk4",
				ReadOnly:  false,
			},
			{
				MountPath: "c:/mnt/path5",
				Name:      "disk5",
				ReadOnly:  false,
			},
			{
				MountPath: `\mnt\path6`,
				Name:      "disk6",
				ReadOnly:  false,
			},
			{
				MountPath: `/mnt/path7`,
				Name:      "disk7",
				ReadOnly:  false,
			},
			{
				MountPath: `\\.\pipe\pipe1`,
				Name:      "pipe1",
				ReadOnly:  false,
			},
		},
	}

	podVolumes := kubecontainer.VolumeMap{
		"disk":  kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "c:/mnt/disk"}},
		"disk4": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "c:/mnt/host"}},
		"disk5": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "c:/var/lib/kubelet/podID/volumes/empty/disk5"}},
		"disk6": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: `/mnt/disk6`}},
		"disk7": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: `\mnt\disk7`}},
		"pipe1": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: `\\.\pipe\pipe1`}},
	}

	pod := v1.Pod{
		Spec: v1.PodSpec{
			HostNetwork: true,
		},
	}

	fhu := &mount.FakeHostUtil{}
	fsp := &subpath.FakeSubpath{}
	mounts, _, _ := makeMounts(&pod, "/pod", &container, "fakepodname", "", "", podVolumes, fhu, fsp, nil)

	expectedMounts := []kubecontainer.Mount{
		{
			Name:           "disk",
			ContainerPath:  "c:/etc/hosts",
			HostPath:       "c:/mnt/disk",
			ReadOnly:       false,
			SELinuxRelabel: false,
		},
		{
			Name:           "disk",
			ContainerPath:  "c:/mnt/path3",
			HostPath:       "c:/mnt/disk",
			ReadOnly:       true,
			SELinuxRelabel: false,
		},
		{
			Name:           "disk4",
			ContainerPath:  "c:/mnt/path4",
			HostPath:       "c:/mnt/host",
			ReadOnly:       false,
			SELinuxRelabel: false,
		},
		{
			Name:           "disk5",
			ContainerPath:  "c:/mnt/path5",
			HostPath:       "c:/var/lib/kubelet/podID/volumes/empty/disk5",
			ReadOnly:       false,
			SELinuxRelabel: false,
		},
		{
			Name:           "disk6",
			ContainerPath:  `c:\mnt\path6`,
			HostPath:       `c:/mnt/disk6`,
			ReadOnly:       false,
			SELinuxRelabel: false,
		},
		{
			Name:           "disk7",
			ContainerPath:  `c:/mnt/path7`,
			HostPath:       `c:\mnt\disk7`,
			ReadOnly:       false,
			SELinuxRelabel: false,
		},
		{
			Name:           "pipe1",
			ContainerPath:  `\\.\pipe\pipe1`,
			HostPath:       `\\.\pipe\pipe1`,
			ReadOnly:       false,
			SELinuxRelabel: false,
		},
	}
	assert.Equal(t, expectedMounts, mounts, "mounts of container %+v", container)
}
