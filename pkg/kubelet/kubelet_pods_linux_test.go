// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	"k8s.io/kubernetes/pkg/volume/util/subpath"
)

func TestMakeMounts(t *testing.T) {
	bTrue := true
	propagationHostToContainer := v1.MountPropagationHostToContainer
	propagationBidirectional := v1.MountPropagationBidirectional
	propagationNone := v1.MountPropagationNone

	testCases := map[string]struct {
		container      v1.Container
		podVolumes     kubecontainer.VolumeMap
		expectErr      bool
		expectedErrMsg string
		expectedMounts []kubecontainer.Mount
	}{
		"valid mounts in unprivileged container": {
			podVolumes: kubecontainer.VolumeMap{
				"disk":  kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
				"disk4": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/host"}},
				"disk5": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/var/lib/kubelet/podID/volumes/empty/disk5"}},
			},
			container: v1.Container{
				Name: "container1",
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath:        "/etc/hosts",
						Name:             "disk",
						ReadOnly:         false,
						MountPropagation: &propagationHostToContainer,
					},
					{
						MountPath:        "/mnt/path3",
						Name:             "disk",
						ReadOnly:         true,
						MountPropagation: &propagationNone,
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
			},
			expectedMounts: []kubecontainer.Mount{
				{
					Name:           "disk",
					ContainerPath:  "/etc/hosts",
					HostPath:       "/mnt/disk",
					ReadOnly:       false,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
				},
				{
					Name:           "disk",
					ContainerPath:  "/mnt/path3",
					HostPath:       "/mnt/disk",
					ReadOnly:       true,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_PRIVATE,
				},
				{
					Name:           "disk4",
					ContainerPath:  "/mnt/path4",
					HostPath:       "/mnt/host",
					ReadOnly:       false,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_PRIVATE,
				},
				{
					Name:           "disk5",
					ContainerPath:  "/mnt/path5",
					HostPath:       "/var/lib/kubelet/podID/volumes/empty/disk5",
					ReadOnly:       false,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_PRIVATE,
				},
			},
			expectErr: false,
		},
		"valid mounts in privileged container": {
			podVolumes: kubecontainer.VolumeMap{
				"disk":  kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
				"disk4": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/host"}},
				"disk5": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/var/lib/kubelet/podID/volumes/empty/disk5"}},
			},
			container: v1.Container{
				Name: "container1",
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath:        "/etc/hosts",
						Name:             "disk",
						ReadOnly:         false,
						MountPropagation: &propagationBidirectional,
					},
					{
						MountPath:        "/mnt/path3",
						Name:             "disk",
						ReadOnly:         true,
						MountPropagation: &propagationHostToContainer,
					},
					{
						MountPath: "/mnt/path4",
						Name:      "disk4",
						ReadOnly:  false,
					},
				},
				SecurityContext: &v1.SecurityContext{
					Privileged: &bTrue,
				},
			},
			expectedMounts: []kubecontainer.Mount{
				{
					Name:           "disk",
					ContainerPath:  "/etc/hosts",
					HostPath:       "/mnt/disk",
					ReadOnly:       false,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_BIDIRECTIONAL,
				},
				{
					Name:           "disk",
					ContainerPath:  "/mnt/path3",
					HostPath:       "/mnt/disk",
					ReadOnly:       true,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
				},
				{
					Name:           "disk4",
					ContainerPath:  "/mnt/path4",
					HostPath:       "/mnt/host",
					ReadOnly:       false,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_PRIVATE,
				},
			},
			expectErr: false,
		},
		"invalid absolute SubPath": {
			podVolumes: kubecontainer.VolumeMap{
				"disk": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
			},
			container: v1.Container{
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath: "/mnt/path3",
						SubPath:   "/must/not/be/absolute",
						Name:      "disk",
						ReadOnly:  true,
					},
				},
			},
			expectErr:      true,
			expectedErrMsg: "error SubPath `/must/not/be/absolute` must not be an absolute path",
		},
		"invalid SubPath with backsteps": {
			podVolumes: kubecontainer.VolumeMap{
				"disk": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
			},
			container: v1.Container{
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath: "/mnt/path3",
						SubPath:   "no/backsteps/../allowed",
						Name:      "disk",
						ReadOnly:  true,
					},
				},
			},
			expectErr:      true,
			expectedErrMsg: "unable to provision SubPath `no/backsteps/../allowed`: must not contain '..'",
		},
		"volume doesn't exist": {
			podVolumes: kubecontainer.VolumeMap{},
			container: v1.Container{
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath: "/mnt/path3",
						Name:      "disk",
						ReadOnly:  true,
					},
				},
			},
			expectErr:      true,
			expectedErrMsg: "cannot find volume \"disk\" to mount into container \"\"",
		},
		"volume mounter is nil": {
			podVolumes: kubecontainer.VolumeMap{
				"disk": kubecontainer.VolumeInfo{},
			},
			container: v1.Container{
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath: "/mnt/path3",
						Name:      "disk",
						ReadOnly:  true,
					},
				},
			},
			expectErr:      true,
			expectedErrMsg: "cannot find volume \"disk\" to mount into container \"\"",
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			fhu := hostutil.NewFakeHostUtil(nil)
			fsp := &subpath.FakeSubpath{}
			pod := v1.Pod{
				Spec: v1.PodSpec{
					HostNetwork: true,
				},
			}

			mounts, _, err := makeMounts(&pod, "/pod", &tc.container, "fakepodname", "", []string{""}, tc.podVolumes, fhu, fsp, nil)

			// validate only the error if we expect an error
			if tc.expectErr {
				if err == nil || err.Error() != tc.expectedErrMsg {
					t.Fatalf("expected error message `%s` but got `%v`", tc.expectedErrMsg, err)
				}
				return
			}

			// otherwise validate the mounts
			if err != nil {
				t.Fatal(err)
			}

			assert.Equal(t, tc.expectedMounts, mounts, "mounts of container %+v", tc.container)
		})
	}
}

func TestMakeBlockVolumes(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	testCases := map[string]struct {
		container       v1.Container
		podVolumes      kubecontainer.VolumeMap
		expectErr       bool
		expectedErrMsg  string
		expectedDevices []kubecontainer.DeviceInfo
	}{
		"valid volumeDevices in container": {
			podVolumes: kubecontainer.VolumeMap{
				"disk1": kubecontainer.VolumeInfo{BlockVolumeMapper: &stubBlockVolume{dirPath: "/dev/", volName: "sda"}},
				"disk2": kubecontainer.VolumeInfo{BlockVolumeMapper: &stubBlockVolume{dirPath: "/dev/disk/by-path/", volName: "diskPath"}, ReadOnly: true},
				"disk3": kubecontainer.VolumeInfo{BlockVolumeMapper: &stubBlockVolume{dirPath: "/dev/disk/by-id/", volName: "diskUuid"}},
				"disk4": kubecontainer.VolumeInfo{BlockVolumeMapper: &stubBlockVolume{dirPath: "/var/lib/", volName: "rawdisk"}, ReadOnly: true},
			},
			container: v1.Container{
				Name: "container1",
				VolumeDevices: []v1.VolumeDevice{
					{
						DevicePath: "/dev/sda",
						Name:       "disk1",
					},
					{
						DevicePath: "/dev/xvda",
						Name:       "disk2",
					},
					{
						DevicePath: "/dev/xvdb",
						Name:       "disk3",
					},
					{
						DevicePath: "/mnt/rawdisk",
						Name:       "disk4",
					},
				},
			},
			expectedDevices: []kubecontainer.DeviceInfo{
				{
					PathInContainer: "/dev/sda",
					PathOnHost:      "/dev/sda",
					Permissions:     "mrw",
				},
				{
					PathInContainer: "/dev/xvda",
					PathOnHost:      "/dev/disk/by-path/diskPath",
					Permissions:     "r",
				},
				{
					PathInContainer: "/dev/xvdb",
					PathOnHost:      "/dev/disk/by-id/diskUuid",
					Permissions:     "mrw",
				},
				{
					PathInContainer: "/mnt/rawdisk",
					PathOnHost:      "/var/lib/rawdisk",
					Permissions:     "r",
				},
			},
			expectErr: false,
		},
		"invalid absolute Path": {
			podVolumes: kubecontainer.VolumeMap{
				"disk": kubecontainer.VolumeInfo{BlockVolumeMapper: &stubBlockVolume{dirPath: "/dev/", volName: "sda"}},
			},
			container: v1.Container{
				VolumeDevices: []v1.VolumeDevice{
					{
						DevicePath: "must/be/absolute",
						Name:       "disk",
					},
				},
			},
			expectErr:      true,
			expectedErrMsg: "error DevicePath `must/be/absolute` must be an absolute path",
		},
		"volume doesn't exist": {
			podVolumes: kubecontainer.VolumeMap{},
			container: v1.Container{
				VolumeDevices: []v1.VolumeDevice{
					{
						DevicePath: "/dev/sdaa",
						Name:       "disk",
					},
				},
			},
			expectErr:      true,
			expectedErrMsg: "cannot find volume \"disk\" to pass into container \"\"",
		},
		"volume BlockVolumeMapper is nil": {
			podVolumes: kubecontainer.VolumeMap{
				"disk": kubecontainer.VolumeInfo{},
			},
			container: v1.Container{
				VolumeDevices: []v1.VolumeDevice{
					{
						DevicePath: "/dev/sdzz",
						Name:       "disk",
					},
				},
			},
			expectErr:      true,
			expectedErrMsg: "cannot find volume \"disk\" to pass into container \"\"",
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			pod := v1.Pod{
				Spec: v1.PodSpec{
					HostNetwork: true,
				},
			}
			blkutil := volumetest.NewBlockVolumePathHandler()
			blkVolumes, err := kubelet.makeBlockVolumes(&pod, &tc.container, tc.podVolumes, blkutil)
			// validate only the error if we expect an error
			if tc.expectErr {
				if err == nil || err.Error() != tc.expectedErrMsg {
					t.Fatalf("expected error message `%s` but got `%v`", tc.expectedErrMsg, err)
				}
				return
			}
			// otherwise validate the devices
			if err != nil {
				t.Fatal(err)
			}
			assert.Equal(t, tc.expectedDevices, blkVolumes, "devices of container %+v", tc.container)
		})
	}
}
