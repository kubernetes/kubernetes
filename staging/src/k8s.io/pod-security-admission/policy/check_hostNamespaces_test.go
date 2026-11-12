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

func TestHostNamespaces(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		expectReason string
		expectDetail string
	}{
		{
			name: "multiple host namespaces",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				HostNetwork: true,
				HostIPC:     true,
				HostPID:     true,
			}},
			expectReason: `host namespaces`,
			expectDetail: `hostNetwork=true, hostPID=true, hostIPC=true`,
		},
		{
			name: "host cgroup namespace",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				InitContainers: []corev1.Container{{Name: "a", HostCgroupNamespace: new(true)}},
				Containers: []corev1.Container{
					{Name: "b", HostCgroupNamespace: new(false)},
					{Name: "c", HostCgroupNamespace: new(true)},
					{Name: "d"},
				},
				EphemeralContainers: []corev1.EphemeralContainer{{EphemeralContainerCommon: corev1.EphemeralContainerCommon{Name: "e", HostCgroupNamespace: new(true)}}},
			}},
			expectReason: `host namespaces`,
			expectDetail: `containers "a", "c", "e" must not set hostCgroupNamespace=true`,
		},
		{
			name: "host network and host cgroup namespace",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				HostNetwork: true,
				Containers:  []corev1.Container{{Name: "a", HostCgroupNamespace: new(true)}},
			}},
			expectReason: `host namespaces`,
			expectDetail: `hostNetwork=true, container "a" must not set hostCgroupNamespace=true`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := hostNamespaces_1_0(&tc.pod.ObjectMeta, &tc.pod.Spec)
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
