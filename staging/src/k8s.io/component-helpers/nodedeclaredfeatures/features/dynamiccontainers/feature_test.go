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

package dynamiccontainers

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestDiscover(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		cfg := &types.NodeConfiguration{FeatureGates: types.FeatureGateMap{DynamicContainersFeatureGate: enabled}}
		if got := Feature.Discover(cfg); got != enabled {
			t.Errorf("Discover() with gate %v = %v", enabled, got)
		}
	}
}

func TestInferForScheduling(t *testing.T) {
	podInfo := &types.PodInfo{Spec: &v1.PodSpec{Containers: []v1.Container{{Name: "c"}}}, Status: &v1.PodStatus{}}
	if Feature.InferForScheduling(podInfo) {
		t.Error("InferForScheduling should always be false")
	}
}

func TestInferForUpdate(t *testing.T) {
	podInfo := func(containers ...v1.Container) *types.PodInfo {
		return &types.PodInfo{Spec: &v1.PodSpec{Containers: containers}, Status: &v1.PodStatus{}}
	}
	c1 := v1.Container{Name: "c1", Image: "image"}
	c2 := v1.Container{Name: "c2", Image: "image"}
	c3 := v1.Container{Name: "c3", Image: "image"}
	c1Updated := v1.Container{Name: "c1", Image: "new-image"}

	tests := []struct {
		name     string
		old, new *types.PodInfo
		expected bool
	}{
		{name: "unchanged", old: podInfo(c1, c2), new: podInfo(c1, c2), expected: false},
		{name: "image updated", old: podInfo(c1, c2), new: podInfo(c1Updated, c2), expected: false},
		{name: "containers reordered", old: podInfo(c1, c2), new: podInfo(c2, c1), expected: true},
		{name: "container added", old: podInfo(c1), new: podInfo(c1, c2), expected: true},
		{name: "container removed", old: podInfo(c1, c2), new: podInfo(c1), expected: true},
		{name: "container added and removed", old: podInfo(c1, c2), new: podInfo(c1, c3), expected: true},
		{
			name: "init container changes are ignored",
			old:  podInfo(c1),
			new: &types.PodInfo{
				Spec:   &v1.PodSpec{Containers: []v1.Container{c1}, InitContainers: []v1.Container{c2}},
				Status: &v1.PodStatus{},
			},
			expected: false,
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
