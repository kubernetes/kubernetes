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

func TestCapabilitiesRestricted(t *testing.T) {
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
					{Name: "a", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"FOO", "BAR"}}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"BAR", "BAZ"}}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"NET_BIND_SERVICE", "CHOWN"}, Drop: []corev1.Capability{"ALL", "FOO"}}}},
				}}},
			expectReason: `unrestricted capabilities`,
			expectDetail: `containers "a", "b" must set securityContext.capabilities.drop=["ALL"]; containers "a", "b", "c" must not include "BAR", "BAZ", "CHOWN", "FOO" in securityContext.capabilities.add`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := capabilitiesRestricted_1_22(&tc.pod.ObjectMeta, &tc.pod.Spec)
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
