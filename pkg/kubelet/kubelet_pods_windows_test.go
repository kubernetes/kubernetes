//go:build windows

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
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	cmtesting "k8s.io/kubernetes/pkg/kubelet/cm/testing"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	"k8s.io/kubernetes/pkg/volume/util/subpath"
)

func TestMakeMountsWindows(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	// TODO: remove skip once the failing test has been fixed.
	t.Skip("Skip failing test on Windows.")
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

	fhu := hostutil.NewFakeHostUtil(nil)
	fsp := &subpath.FakeSubpath{}
	podDir, err := os.MkdirTemp("", "test-rotate-logs")
	require.NoError(t, err)
	defer os.RemoveAll(podDir)
	mounts, _, err := makeMounts(logger, &pod, podDir, &container, "fakepodname", "", []string{""}, podVolumes, fhu, fsp, nil, false, nil)
	require.NoError(t, err)

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
		{
			Name:           "k8s-managed-etc-hosts",
			ContainerPath:  `C:\Windows\System32\drivers\etc\hosts`,
			HostPath:       filepath.Join(podDir, "etc-hosts"),
			ReadOnly:       false,
			SELinuxRelabel: true,
		},
	}
	assert.Equal(t, expectedMounts, mounts, "mounts of container %+v", container)
}

// TestConvertToAPIPodLevelResourcesStatus verifies that on Windows, where
// in-place pod-level resize is not supported and pods are not backed by
// cgroups, the kubelet reports the allocated pod-level resources without
// consulting the pod container manager at all.
//
// See https://github.com/kubernetes/kubernetes/issues/141760
func TestConvertToAPIPodLevelResourcesStatus(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.PodLevelResources:                       true,
		features.InPlacePodLevelResourcesVerticalScaling: true,
	})

	testCases := []struct {
		name     string
		pod      *v1.Pod
		expected *v1.ResourceRequirements
	}{
		{
			name: "pod-level resources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("50m"),
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
					},
					Containers: []v1.Container{{Name: "pause"}},
				},
				Status: v1.PodStatus{Phase: v1.PodRunning},
			},
			expected: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("50m"),
					v1.ResourceMemory: resource.MustParse("50Mi"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
			},
		},
		{
			name: "container-level resources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("50m"),
									v1.ResourceMemory: resource.MustParse("50Mi"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
				Status: v1.PodStatus{Phase: v1.PodRunning},
			},
			expected: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("50m"),
					v1.ResourceMemory: resource.MustParse("50Mi"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
			},
		},
	}

	// Synthetic oldPodStatus status whose values deliberately do not match the pod
	// spec.
	//
	// Before the platform guard, the failed cgroup read left every readback nil
	// and preserveOldResourcesValue copied the previous status into the result.
	// With the guard, it now returns early from the allocated spec and
	// the previous status is never consulted, so these values must not appear.
	oldPodStatus := v1.PodStatus{
		Phase: v1.PodRunning,
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("999m")},
			Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("999m"), v1.ResourceMemory: resource.MustParse("999Mi")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			mockCM := cmtesting.NewMockContainerManager(t)

			kl := &Kubelet{containerManager: mockCM}
			got := kl.convertToAPIPodLevelResourcesStatus(logger, tc.pod, oldPodStatus, tc.pod.Status.Phase)
			require.NotNil(t, got)

			// Result must be the allocated (spec) resources, not the stale/synthetic status.
			require.Equal(t, tc.expected, got)
		})
	}
}
