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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestSeccompProfileBaseline_1_0(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		expectReason string
		expectDetail string
	}{
		{
			name: "pod seccomp invalid",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"seccomp.security.alpha.kubernetes.io/pod": "unconfined",
					},
				},
			},
			expectReason: `seccompProfile`,
			expectDetail: `forbidden annotation seccomp.security.alpha.kubernetes.io/pod="unconfined"`,
		},
		{
			name: "containers seccomp invalid",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"container.seccomp.security.alpha.kubernetes.io/a": "unconfined",
						"container.seccomp.security.alpha.kubernetes.io/b": "unknown",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "a"}, {Name: "b"}, {Name: "c"}},
				},
			},
			expectReason: `seccompProfile`,
			expectDetail: `forbidden annotations container.seccomp.security.alpha.kubernetes.io/a="unconfined", container.seccomp.security.alpha.kubernetes.io/b="unknown"`,
		},
		{
			name: "pod and containers seccomp invalid",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"seccomp.security.alpha.kubernetes.io/pod":         "unconfined",
						"container.seccomp.security.alpha.kubernetes.io/a": "unconfined",
						"container.seccomp.security.alpha.kubernetes.io/b": "unknown",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "a"}, {Name: "b"}, {Name: "c"}},
				},
			},
			expectReason: `seccompProfile`,
			expectDetail: `forbidden annotations container.seccomp.security.alpha.kubernetes.io/a="unconfined", container.seccomp.security.alpha.kubernetes.io/b="unknown", seccomp.security.alpha.kubernetes.io/pod="unconfined"`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := seccompProfileBaseline_1_0(&tc.pod.ObjectMeta, &tc.pod.Spec)
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

func TestSeccompProfileBaseline_1_19(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		expectReason string
		expectDetail string
	}{
		{
			name: "pod seccomp invalid",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `pod must not set securityContext.seccompProfile.type to "Unconfined"`,
		},
		{
			name: "containers seccomp invalid",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `containers "c", "d" must not set securityContext.seccompProfile.type to "Unconfined"`,
		},
		{
			name: "pod and containers seccomp invalid",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `pod and containers "c", "d" must not set securityContext.seccompProfile.type to "Unconfined"`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := seccompProfileBaseline_1_19(&tc.pod.ObjectMeta, &tc.pod.Spec)
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
