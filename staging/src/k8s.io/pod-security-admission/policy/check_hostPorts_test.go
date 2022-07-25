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

func TestHostPort(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		expectReason string
		expectDetail string
	}{
		{
			name: "one container, one host port",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", Ports: []corev1.ContainerPort{{HostPort: 0}}},
					{Name: "b", Ports: []corev1.ContainerPort{{HostPort: 0}, {HostPort: 20}}},
				},
			}},
			expectReason: `hostPort`,
			expectDetail: `container "b" uses hostPort 20`,
		},
		{
			name: "multiple containers, multiple host port",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", Ports: []corev1.ContainerPort{{HostPort: 0}}},
					{Name: "b", Ports: []corev1.ContainerPort{{HostPort: 0}, {HostPort: 10}, {HostPort: 20}}},
					{Name: "c", Ports: []corev1.ContainerPort{{HostPort: 0}, {HostPort: 10}, {HostPort: 30}}},
				},
			}},
			expectReason: `hostPort`,
			expectDetail: `containers "b", "c" use hostPorts 10, 20, 30`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := hostPorts_1_0(&tc.pod.ObjectMeta, &tc.pod.Spec)
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
