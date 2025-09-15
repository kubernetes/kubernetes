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
	"k8s.io/utils/ptr"
)

func TestRunAsUser(t *testing.T) {
	tests := []struct {
		name           string
		pod            *corev1.Pod
		expectAllowed  bool
		expectReason   string
		expectDetail   string
		relaxForUserNS bool
	}{
		{
			name: "pod runAsUser=0",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: ptr.To[int64](0)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			expectReason: `runAsUser=0`,
			expectDetail: `pod must not set runAsUser=0`,
		},
		{
			name: "pod runAsUser=non-zero",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: ptr.To[int64](1000)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			expectAllowed: true,
		},
		{
			name: "pod runAsUser=nil",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: nil},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			expectAllowed: true,
		},
		{
			name: "containers runAsUser=0",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: ptr.To[int64](1000)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{RunAsUser: ptr.To[int64](0)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{RunAsUser: ptr.To[int64](0)}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{RunAsUser: ptr.To[int64](1)}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{RunAsUser: ptr.To[int64](1)}},
				},
			}},
			expectReason: `runAsUser=0`,
			expectDetail: `containers "c", "d" must not set runAsUser=0`,
		},
		{
			name: "containers runAsUser=non-zero",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "c", SecurityContext: &corev1.SecurityContext{RunAsUser: ptr.To[int64](1)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{RunAsUser: ptr.To[int64](2)}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{RunAsUser: ptr.To[int64](3)}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{RunAsUser: ptr.To[int64](4)}},
				},
			}},
			expectAllowed: true,
		},
		{
			name: "UserNamespacesPodSecurityStandards enabled without HostUsers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				HostUsers: ptr.To(false),
			}},
			expectAllowed:  true,
			relaxForUserNS: true,
		},
		{
			name: "UserNamespacesPodSecurityStandards enabled with HostUsers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: ptr.To[int64](0)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
				HostUsers: ptr.To(true),
			}},
			expectAllowed:  false,
			expectReason:   `runAsUser=0`,
			expectDetail:   `pod must not set runAsUser=0`,
			relaxForUserNS: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.relaxForUserNS {
				RelaxPolicyForUserNamespacePods(true)
				t.Cleanup(func() {
					RelaxPolicyForUserNamespacePods(false)
				})
			}
			result := runAsUser_1_23(&tc.pod.ObjectMeta, &tc.pod.Spec)
			if result.Allowed != tc.expectAllowed {
				t.Fatalf("expected Allowed to be %v was %v", tc.expectAllowed, result.Allowed)
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
