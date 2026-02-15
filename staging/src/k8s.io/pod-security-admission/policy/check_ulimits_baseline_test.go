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

package policy

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestUlimitsBaseline_1_0(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		expectReason string
		expectDetail string
		allowed      bool
	}{
		{
			name: "multiple containers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
					{Name: "b", SecurityContext: &corev1.SecurityContext{Ulimits: []corev1.Ulimit{{Name: "nofile", Soft: 1024, Hard: 2048}}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{Ulimits: []corev1.Ulimit{{Name: "memlock", Soft: 1024, Hard: 2048}}}},
				},
			}},
			expectReason: `ulimits`,
			expectDetail: `containers "b", "c" must not set securityContext.ulimits`,
			allowed:      false,
		},
		{
			name: "allow when unset",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
				},
			}},
			allowed: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := ulimitsBaseline_1_0(&tc.pod.ObjectMeta, &tc.pod.Spec)
			if result.Allowed && !tc.allowed {
				t.Fatal("expected disallowed")
			}
			if !result.Allowed && tc.allowed {
				t.Fatal("expected allowed")
			}
			if e, a := tc.expectReason, result.ForbiddenReason; e != a {
				t.Errorf("expected\n%s\ngot\n%s", e, a)
			}
			if e, a := tc.expectDetail, result.ForbiddenDetail; e != a {
				t.Errorf("expected\n%s\ngot\n%s", e, a)
			}
		})
	}
}
