/*
Copyright 2015 The Kubernetes Authors.

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

	"fmt"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestHasHostMountPVC(t *testing.T) {
	tests := map[string]struct {
		pvError       error
		pvcError      error
		expected      bool
		podHasPVC     bool
		pvcIsHostPath bool
	}{
		"no pvc": {podHasPVC: false, expected: false},
		"error fetching pvc": {
			podHasPVC: true,
			pvcError:  fmt.Errorf("foo"),
			expected:  false,
		},
		"error fetching pv": {
			podHasPVC: true,
			pvError:   fmt.Errorf("foo"),
			expected:  false,
		},
		"host path pvc": {
			podHasPVC:     true,
			pvcIsHostPath: true,
			expected:      true,
		},
		"non host path pvc": {
			podHasPVC:     true,
			pvcIsHostPath: false,
			expected:      false,
		},
	}

	for k, v := range tests {
		testKubelet := newTestKubelet(t, false)
		pod := &api.Pod{
			Spec: api.PodSpec{},
		}

		volumeToReturn := &api.PersistentVolume{
			Spec: api.PersistentVolumeSpec{},
		}

		if v.podHasPVC {
			pod.Spec.Volumes = []api.Volume{
				{
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{},
					},
				},
			}

			if v.pvcIsHostPath {
				volumeToReturn.Spec.PersistentVolumeSource = api.PersistentVolumeSource{
					HostPath: &api.HostPathVolumeSource{},
				}
			}

		}

		testKubelet.fakeKubeClient.AddReactor("get", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
			return true, &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "foo",
				},
			}, v.pvcError
		})
		testKubelet.fakeKubeClient.AddReactor("get", "persistentvolumes", func(action core.Action) (bool, runtime.Object, error) {
			return true, volumeToReturn, v.pvError
		})

		actual := testKubelet.kubelet.hasHostMountPVC(pod)
		if actual != v.expected {
			t.Errorf("%s expected %t but got %t", k, v.expected, actual)
		}

	}
}

func TestHasNonNamespacedCapability(t *testing.T) {
	createPodWithCap := func(caps []api.Capability) *api.Pod {
		pod := &api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{{}},
			},
		}

		if len(caps) > 0 {
			pod.Spec.Containers[0].SecurityContext = &api.SecurityContext{
				Capabilities: &api.Capabilities{
					Add: caps,
				},
			}
		}
		return pod
	}

	nilCaps := createPodWithCap([]api.Capability{api.Capability("foo")})
	nilCaps.Spec.Containers[0].SecurityContext = nil

	tests := map[string]struct {
		pod      *api.Pod
		expected bool
	}{
		"nil security contxt":           {createPodWithCap(nil), false},
		"nil caps":                      {nilCaps, false},
		"namespaced cap":                {createPodWithCap([]api.Capability{api.Capability("foo")}), false},
		"non-namespaced cap MKNOD":      {createPodWithCap([]api.Capability{api.Capability("MKNOD")}), true},
		"non-namespaced cap SYS_TIME":   {createPodWithCap([]api.Capability{api.Capability("SYS_TIME")}), true},
		"non-namespaced cap SYS_MODULE": {createPodWithCap([]api.Capability{api.Capability("SYS_MODULE")}), true},
	}

	for k, v := range tests {
		actual := hasNonNamespacedCapability(v.pod)
		if actual != v.expected {
			t.Errorf("%s failed, expected %t but got %t", k, v.expected, actual)
		}
	}
}

func TestHasHostVolume(t *testing.T) {
	pod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{},
					},
				},
			},
		},
	}

	result := hasHostVolume(pod)
	if !result {
		t.Errorf("expected host volume to enable host user namespace")
	}

	pod.Spec.Volumes[0].VolumeSource.HostPath = nil
	result = hasHostVolume(pod)
	if result {
		t.Errorf("expected nil host volume to not enable host user namespace")
	}
}

func TestHasHostNamespace(t *testing.T) {
	tests := map[string]struct {
		psc      *api.PodSecurityContext
		expected bool
	}{
		"nil psc": {psc: nil, expected: false},
		"host pid true": {
			psc: &api.PodSecurityContext{
				HostPID: true,
			},
			expected: true,
		},
		"host ipc true": {
			psc: &api.PodSecurityContext{
				HostIPC: true,
			},
			expected: true,
		},
		"host net true": {
			psc: &api.PodSecurityContext{
				HostNetwork: true,
			},
			expected: true,
		},
		"no host ns": {
			psc:      &api.PodSecurityContext{},
			expected: false,
		},
	}

	for k, v := range tests {
		pod := &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: v.psc,
			},
		}
		actual := hasHostNamespace(pod)
		if actual != v.expected {
			t.Errorf("%s failed, expected %t but got %t", k, v.expected, actual)
		}
	}
}
