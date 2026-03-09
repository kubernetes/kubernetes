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

func TestCgroupOptionsRestricted(t *testing.T) {
	readOnly := corev1.CgroupMountModeReadOnly
	writable := corev1.CgroupMountModeWritable

	tests := []struct {
		name          string
		pod           *corev1.Pod
		expectReason  string
		expectDetail  string
		expectAllowed bool
	}{
		{
			name: "no cgroupOptions",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{CgroupOptions: &corev1.CgroupOptions{}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{CgroupOptions: &corev1.CgroupOptions{MountMode: &readOnly}}},
				},
			}},
			expectAllowed: true,
		},
		{
			name: "writable cgroups",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{CgroupOptions: &corev1.CgroupOptions{MountMode: &readOnly}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{CgroupOptions: &corev1.CgroupOptions{MountMode: &writable}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{CgroupOptions: &corev1.CgroupOptions{MountMode: &writable}}},
				},
			}},
			expectReason:  `cgroupOptions`,
			expectAllowed: false,
			expectDetail:  `containers "b", "c" must not set securityContext.cgroupOptions.mountMode to "Writable"`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := cgroupOptionsRestricted1_38(&tc.pod.ObjectMeta, &tc.pod.Spec)
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
