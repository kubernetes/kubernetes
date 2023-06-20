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

func TestRunAsNonRoot(t *testing.T) {
	tests := []struct {
		name                                     string
		pod                                      *corev1.Pod
		expectReason                             string
		expectDetail                             string
		allowed                                  bool
		enableUserNamespacesPodSecurityStandards bool
	}{
		{
			name: "no explicit runAsNonRoot",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
				},
			}},
			expectReason: `runAsNonRoot != true`,
			expectDetail: `pod or container "a" must set securityContext.runAsNonRoot=true`,
		},
		{
			name: "pod runAsNonRoot=false",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsNonRoot: utilpointer.Bool(false)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			expectReason: `runAsNonRoot != true`,
			expectDetail: `pod must not set securityContext.runAsNonRoot=false`,
		},
		{
			name: "containers runAsNonRoot=false",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsNonRoot: utilpointer.Bool(true)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{RunAsNonRoot: utilpointer.Bool(false)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{RunAsNonRoot: utilpointer.Bool(false)}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{RunAsNonRoot: utilpointer.Bool(true)}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{RunAsNonRoot: utilpointer.Bool(true)}},
				},
			}},
			expectReason: `runAsNonRoot != true`,
			expectDetail: `containers "c", "d" must not set securityContext.runAsNonRoot=false`,
		},
		{
			name: "pod nil, container fallthrough",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{RunAsNonRoot: utilpointer.Bool(true)}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{RunAsNonRoot: utilpointer.Bool(true)}},
				},
			}},
			expectReason: `runAsNonRoot != true`,
			expectDetail: `pod or containers "a", "b" must set securityContext.runAsNonRoot=true`,
		},
		{
			name: "UserNamespacesPodSecurityStandards enabled without HostUsers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				HostUsers: utilpointer.Bool(false),
			}},
			allowed:                                  true,
			enableUserNamespacesPodSecurityStandards: true,
		},
		{
			name: "UserNamespacesPodSecurityStandards enabled with HostUsers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
				},
				HostUsers: utilpointer.Bool(true),
			}},
			expectReason:                             `runAsNonRoot != true`,
			expectDetail:                             `pod or container "a" must set securityContext.runAsNonRoot=true`,
			allowed:                                  false,
			enableUserNamespacesPodSecurityStandards: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.enableUserNamespacesPodSecurityStandards {
				RelaxPolicyForUserNamespacePods(true)
			}
			result := runAsNonRoot_1_0(&tc.pod.ObjectMeta, &tc.pod.Spec)
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
