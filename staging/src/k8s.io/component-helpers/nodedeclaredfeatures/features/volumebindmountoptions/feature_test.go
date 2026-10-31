/*
Copyright 2026 The Kubernetes Authors.

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

package volumebindmountoptions

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func mountWithOptions(name string, options ...string) v1.VolumeMount {
	return v1.VolumeMount{Name: name, MountPath: "/" + name, BindMountOptions: options}
}

func podSpecWithEphemeral(ephemeral ...v1.EphemeralContainer) *v1.PodSpec {
	return &v1.PodSpec{
		Containers:          []v1.Container{{Name: "app", Image: "image"}},
		EphemeralContainers: ephemeral,
	}
}

func ephemeralContainer(name string, mounts ...v1.VolumeMount) v1.EphemeralContainer {
	return v1.EphemeralContainer{EphemeralContainerCommon: v1.EphemeralContainerCommon{
		Name:         name,
		Image:        "debug",
		VolumeMounts: mounts,
	}}
}

func TestInferForScheduling(t *testing.T) {
	tests := []struct {
		name     string
		spec     *v1.PodSpec
		expected bool
	}{
		{
			name:     "no bind mount options",
			spec:     &v1.PodSpec{Containers: []v1.Container{{Name: "app", VolumeMounts: []v1.VolumeMount{mountWithOptions("data")}}}},
			expected: false,
		},
		{
			name:     "container",
			spec:     &v1.PodSpec{Containers: []v1.Container{{Name: "app", VolumeMounts: []v1.VolumeMount{mountWithOptions("data", "noexec")}}}},
			expected: true,
		},
		{
			name:     "init container",
			spec:     &v1.PodSpec{InitContainers: []v1.Container{{Name: "init", VolumeMounts: []v1.VolumeMount{mountWithOptions("data", "nosuid")}}}},
			expected: true,
		},
		{
			name:     "ephemeral container",
			spec:     podSpecWithEphemeral(ephemeralContainer("debugger", mountWithOptions("data", "nodev"))),
			expected: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := Feature.InferForScheduling(&types.PodInfo{Spec: tc.spec}); got != tc.expected {
				t.Errorf("InferForScheduling() = %v, want %v", got, tc.expected)
			}
		})
	}
}

func TestInferForUpdate(t *testing.T) {
	tests := []struct {
		name     string
		oldSpec  *v1.PodSpec
		newSpec  *v1.PodSpec
		expected bool
	}{
		{
			name:     "no change",
			oldSpec:  podSpecWithEphemeral(),
			newSpec:  podSpecWithEphemeral(),
			expected: false,
		},
		{
			name:     "ephemeral container appended without bind mount options",
			oldSpec:  podSpecWithEphemeral(),
			newSpec:  podSpecWithEphemeral(ephemeralContainer("debugger", mountWithOptions("data"))),
			expected: false,
		},
		{
			name:     "ephemeral container appended with bind mount options",
			oldSpec:  podSpecWithEphemeral(),
			newSpec:  podSpecWithEphemeral(ephemeralContainer("debugger", mountWithOptions("data", "noexec"))),
			expected: true,
		},
		{
			name:     "second ephemeral container appended with bind mount options",
			oldSpec:  podSpecWithEphemeral(ephemeralContainer("first")),
			newSpec:  podSpecWithEphemeral(ephemeralContainer("first"), ephemeralContainer("second", mountWithOptions("data", "nosuid"))),
			expected: true,
		},
		{
			name:     "pre-existing ephemeral container is not re-reported",
			oldSpec:  podSpecWithEphemeral(ephemeralContainer("debugger", mountWithOptions("data", "noexec"))),
			newSpec:  podSpecWithEphemeral(ephemeralContainer("debugger", mountWithOptions("data", "noexec"))),
			expected: false,
		},
		{
			// Order is not preserved by patch, so the added container may appear before an existing one.
			name:    "added container with bind mount options placed before an existing one",
			oldSpec: podSpecWithEphemeral(ephemeralContainer("existing")),
			newSpec: podSpecWithEphemeral(
				ephemeralContainer("debugger", mountWithOptions("data", "noexec")),
				ephemeralContainer("existing"),
			),
			expected: true,
		},
		{
			name: "existing option-bearing container follows a new container without options",
			oldSpec: podSpecWithEphemeral(
				ephemeralContainer("secured", mountWithOptions("data", "noexec")),
				ephemeralContainer("existing"),
			),
			newSpec: podSpecWithEphemeral(
				ephemeralContainer("existing"),
				ephemeralContainer("debugger", mountWithOptions("data")),
				ephemeralContainer("secured", mountWithOptions("data", "noexec")),
			),
			expected: false,
		},
		{
			// Only the reordering is new here; the option-bearing container already existed.
			name: "existing container with bind mount options reordered",
			oldSpec: podSpecWithEphemeral(
				ephemeralContainer("secured", mountWithOptions("data", "noexec")),
				ephemeralContainer("existing"),
			),
			newSpec: podSpecWithEphemeral(
				ephemeralContainer("existing"),
				ephemeralContainer("secured", mountWithOptions("data", "noexec")),
			),
			expected: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Feature.InferForUpdate(&types.PodInfo{Spec: tc.oldSpec}, &types.PodInfo{Spec: tc.newSpec})
			if got != tc.expected {
				t.Errorf("InferForUpdate() = %v, want %v", got, tc.expected)
			}
		})
	}
}
