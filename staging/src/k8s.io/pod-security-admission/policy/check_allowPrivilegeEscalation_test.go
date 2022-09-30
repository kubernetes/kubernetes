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
	utilpointer "k8s.io/utils/pointer"
)

func TestAllowPrivilegeEscalation_1_25(t *testing.T) {
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
					{Name: "b", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: nil}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(true)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(false)}},
				}}},
			expectReason: `allowPrivilegeEscalation != false`,
			expectDetail: `containers "a", "b", "c" must set securityContext.allowPrivilegeEscalation=false`,
			allowed:      false,
		},
		{
			name: "windows pod, admit without checking privilegeEscalation",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				OS: &corev1.PodOS{Name: corev1.Windows},
				Containers: []corev1.Container{
					{Name: "a"},
				}}},
			allowed: true,
		},
		{
			name: "linux pod, reject if security context is not set",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				OS: &corev1.PodOS{Name: corev1.Linux},
				Containers: []corev1.Container{
					{Name: "a"},
				}}},
			expectReason: `allowPrivilegeEscalation != false`,
			expectDetail: `container "a" must set securityContext.allowPrivilegeEscalation=false`,
			allowed:      false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := allowPrivilegeEscalation_1_25(&tc.pod.ObjectMeta, &tc.pod.Spec)
			if result.Allowed && !tc.allowed {
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

func TestAllowPrivilegeEscalation_1_8(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		expectReason string
		expectDetail string
	}{
		{
			name: "multiple containers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
					{Name: "b", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: nil}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(true)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(false)}},
				}}},
			expectReason: `allowPrivilegeEscalation != false`,
			expectDetail: `containers "a", "b", "c" must set securityContext.allowPrivilegeEscalation=false`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := allowPrivilegeEscalation_1_8(&tc.pod.ObjectMeta, &tc.pod.Spec)
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
