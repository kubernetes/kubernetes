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

package scheduling

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corev1 "k8s.io/api/core/v1"
)

func TestIsKnownSystemPriorityClass(t *testing.T) {
	tests := []struct {
		name     string
		pc       *PriorityClass
		expected bool
	}{
		{
			name:     "system priority class",
			pc:       SystemPriorityClasses()[0],
			expected: true,
		},
		{
			name: "non-system priority class",
			pc: &PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: SystemNodeCritical,
				},
				Value:       SystemCriticalPriority, // This is the value of system cluster critical
				Description: "Used for system critical pods that must not be moved from their current node.",
			},
			expected: false,
		},
	}

	for _, test := range tests {
		if is, err := IsKnownSystemPriorityClass(test.pc); test.expected != is {
			t.Errorf("Test [%v]: Expected %v, but got %v. Error: %v", test.name, test.expected, is, err)
		}
	}
}

// TestGetPodPriority tests GetPodPriority function.
func TestGetPodPriority(t *testing.T) {
	p := int32(20)
	tests := []struct {
		name             string
		pod              *corev1.Pod
		expectedPriority int32
	}{
		{
			name: "no priority pod resolves to static default priority",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{Containers: []corev1.Container{
					{Name: "container", Image: "image"}},
				},
			},
			expectedPriority: DefaultPriorityWhenNoDefaultClassExists,
		},
		{
			name: "pod with priority resolves correctly",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{Containers: []corev1.Container{
					{Name: "container", Image: "image"}},
					Priority: &p,
				},
			},
			expectedPriority: p,
		},
	}
	for _, test := range tests {
		if GetPodPriority(test.pod) != test.expectedPriority {
			t.Errorf("expected pod priority: %v, got %v", test.expectedPriority, GetPodPriority(test.pod))
		}

	}
}
