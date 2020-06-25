/*
Copyright 2020 The Kubernetes Authors.

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

package corev1

import (
	"testing"

	v1 "k8s.io/api/core/v1"
)

// TestPodPriority tests PodPriority function.
func TestPodPriority(t *testing.T) {
	p := int32(20)
	tests := []struct {
		name             string
		pod              *v1.Pod
		expectedPriority int32
	}{
		{
			name: "no priority pod resolves to static default priority",
			pod: &v1.Pod{
				Spec: v1.PodSpec{Containers: []v1.Container{
					{Name: "container", Image: "image"}},
				},
			},
			expectedPriority: 0,
		},
		{
			name: "pod with priority resolves correctly",
			pod: &v1.Pod{
				Spec: v1.PodSpec{Containers: []v1.Container{
					{Name: "container", Image: "image"}},
					Priority: &p,
				},
			},
			expectedPriority: p,
		},
	}
	for _, test := range tests {
		if PodPriority(test.pod) != test.expectedPriority {
			t.Errorf("expected pod priority: %v, got %v", test.expectedPriority, PodPriority(test.pod))
		}

	}
}
