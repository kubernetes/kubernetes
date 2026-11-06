/*
Copyright The Kubernetes Authors.

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

package dynamicvolumes

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestDiscover(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		cfg := &types.NodeConfiguration{FeatureGates: types.FeatureGateMap{DynamicNodeLocalEphemeralVolumesFeatureGate: enabled}}
		if got := Feature.Discover(cfg); got != enabled {
			t.Errorf("Discover() with gate %v = %v", enabled, got)
		}
	}
}

func TestInferForScheduling(t *testing.T) {
	podInfo := &types.PodInfo{
		Spec:   &v1.PodSpec{Volumes: []v1.Volume{{Name: "vol-1"}}, Containers: []v1.Container{{Name: "c1"}}},
		Status: &v1.PodStatus{},
	}
	if Feature.InferForScheduling(podInfo) {
		t.Error("InferForScheduling should always be false")
	}
}

func TestInferForUpdate(t *testing.T) {
	rroDisabled1 := v1.RecursiveReadOnlyDisabled
	rroDisabled2 := v1.RecursiveReadOnlyDisabled
	qty100Mi := resource.MustParse("100Mi")
	qty200Mi := resource.MustParse("200Mi")

	podInfo := func(vols []v1.Volume, ctrs ...v1.Container) *types.PodInfo {
		return &types.PodInfo{
			Spec:   &v1.PodSpec{Volumes: vols, Containers: ctrs},
			Status: &v1.PodStatus{},
		}
	}

	vol1 := v1.Volume{Name: "vol-1", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}
	vol2 := v1.Volume{Name: "vol-2", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}
	mnt1 := v1.VolumeMount{Name: "vol-1", MountPath: "/mnt/vol-1", ReadOnly: true, RecursiveReadOnly: &rroDisabled1}
	mnt1SameValue := v1.VolumeMount{Name: "vol-1", MountPath: "/mnt/vol-1", ReadOnly: true, RecursiveReadOnly: &rroDisabled2}
	mnt2 := v1.VolumeMount{Name: "vol-2", MountPath: "/mnt/vol-2"}

	tests := []struct {
		name     string
		old, new *types.PodInfo
		expected bool
	}{
		{
			name:     "unchanged",
			old:      podInfo([]v1.Volume{vol1}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1}}),
			new:      podInfo([]v1.Volume{vol1}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1SameValue}}),
			expected: false,
		},
		{
			name:     "nil vs empty volumeMounts",
			old:      podInfo([]v1.Volume{vol1}, v1.Container{Name: "c1", VolumeMounts: nil}),
			new:      podInfo([]v1.Volume{vol1}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{}}),
			expected: false,
		},
		{
			name: "memory-backed emptyDir sizeLimit resize is not a dynamic volume mutation",
			old: podInfo([]v1.Volume{{
				Name:         "mem-vol",
				VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory, SizeLimit: &qty100Mi}},
			}}, v1.Container{Name: "c1"}),
			new: podInfo([]v1.Volume{{
				Name:         "mem-vol",
				VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory, SizeLimit: &qty200Mi}},
			}}, v1.Container{Name: "c1"}),
			expected: false,
		},
		{
			name:     "volume added",
			old:      podInfo([]v1.Volume{vol1}),
			new:      podInfo([]v1.Volume{vol1, vol2}),
			expected: true,
		},
		{
			name:     "volume removed",
			old:      podInfo([]v1.Volume{vol1, vol2}),
			new:      podInfo([]v1.Volume{vol1}),
			expected: true,
		},
		{
			name:     "volume replaced at same count",
			old:      podInfo([]v1.Volume{vol1}),
			new:      podInfo([]v1.Volume{vol2}),
			expected: true,
		},
		{
			name:     "volumeMount added to existing container",
			old:      podInfo([]v1.Volume{vol1, vol2}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1}}),
			new:      podInfo([]v1.Volume{vol1, vol2}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1, mnt2}}),
			expected: true,
		},
		{
			name:     "volumeMount removed from existing container",
			old:      podInfo([]v1.Volume{vol1, vol2}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1, mnt2}}),
			new:      podInfo([]v1.Volume{vol1, vol2}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1}}),
			expected: true,
		},
		{
			name:     "added container mounting existing volume does not require DynamicNodeLocalEphemeralVolumes",
			old:      podInfo([]v1.Volume{vol1}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1}}),
			new:      podInfo([]v1.Volume{vol1}, v1.Container{Name: "c-added", VolumeMounts: []v1.VolumeMount{mnt1}}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1}}),
			expected: false,
		},
		{
			name:     "removed container with existing volumeMount does not require DynamicNodeLocalEphemeralVolumes",
			old:      podInfo([]v1.Volume{vol1}, v1.Container{Name: "c-removed", VolumeMounts: []v1.VolumeMount{mnt1}}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1}}),
			new:      podInfo([]v1.Volume{vol1}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1}}),
			expected: false,
		},
		{
			name:     "added container plus mutated volumeMount on existing container requires DynamicNodeLocalEphemeralVolumes",
			old:      podInfo([]v1.Volume{vol1, vol2}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1}}),
			new:      podInfo([]v1.Volume{vol1, vol2}, v1.Container{Name: "c-added"}, v1.Container{Name: "c1", VolumeMounts: []v1.VolumeMount{mnt1, mnt2}}),
			expected: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := Feature.InferForUpdate(tc.old, tc.new); got != tc.expected {
				t.Errorf("InferForUpdate() = %v, want %v", got, tc.expected)
			}
		})
	}
}
