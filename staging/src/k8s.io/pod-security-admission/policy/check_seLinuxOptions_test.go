/*
Copyright 2021 The Kubernetes Authors.

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

func TestSELinuxOptions(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		expectReason string
		expectDetail string
	}{
		{
			name: "invalid pod and containers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						Type: "foo",
						User: "bar",
						Role: "baz",
					},
				},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_t",
					}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_init_t",
					}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_kvm_t",
					}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "bar",
					}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						User: "bar",
					}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Role: "baz",
					}}},
				},
			}},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod and containers "d", "e", "f" set forbidden securityContext.seLinuxOptions: types "bar", "foo"; user may not be set; role may not be set`,
		},
		{
			name: "invalid pod",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{
						Type: "foo",
						User: "bar",
						Role: "baz",
					},
				},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_t",
					}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_init_t",
					}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_kvm_t",
					}}},
				},
			}},
			expectReason: `seLinuxOptions`,
			expectDetail: `pod set forbidden securityContext.seLinuxOptions: type "foo"; user may not be set; role may not be set`,
		},
		{
			name: "invalid containers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					SELinuxOptions: &corev1.SELinuxOptions{},
				},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_t",
					}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_init_t",
					}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "container_kvm_t",
					}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Type: "bar",
					}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						User: "bar",
					}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SELinuxOptions: &corev1.SELinuxOptions{
						Role: "baz",
					}}},
				},
			}},
			expectReason: `seLinuxOptions`,
			expectDetail: `containers "d", "e", "f" set forbidden securityContext.seLinuxOptions: type "bar"; user may not be set; role may not be set`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := seLinuxOptions_1_0(&tc.pod.ObjectMeta, &tc.pod.Spec)
			if result.Allowed {
				t.Fatal("expected disallowed")
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
