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
	"k8s.io/apimachinery/pkg/util/validation/field"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestHostPort(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		opts          options
		expectReason  string
		expectDetail  string
		expectErrList field.ErrorList
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
			name: "one container, one host port, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", Ports: []corev1.ContainerPort{{HostPort: 0}}},
					{Name: "b", Ports: []corev1.ContainerPort{{HostPort: 0}, {HostPort: 20}}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `hostPort`,
			expectDetail: `container "b" uses hostPort 20`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[1].ports[1].hostPort", BadValue: 20},
			},
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
		{
			name: "multiple containers, multiple host port, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", Ports: []corev1.ContainerPort{{HostPort: 0}}},
					{Name: "b", Ports: []corev1.ContainerPort{{HostPort: 0}, {HostPort: 10}, {HostPort: 20}}},
					{Name: "c", Ports: []corev1.ContainerPort{{HostPort: 0}, {HostPort: 10}, {HostPort: 30}}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `hostPort`,
			expectDetail: `containers "b", "c" use hostPorts 10, 20, 30`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[1].ports[1].hostPort", BadValue: 10},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[1].ports[2].hostPort", BadValue: 20},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[2].ports[1].hostPort", BadValue: 10},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[2].ports[2].hostPort", BadValue: 30},
			},
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := hostPortsV1Dot0(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
			if result.Allowed {
				t.Fatal("expected disallowed")
			}
			if e, a := tc.expectReason, result.ForbiddenReason; e != a {
				t.Errorf("expected\n%s\ngot\n%s", e, a)
			}
			if e, a := tc.expectDetail, result.ForbiddenDetail; e != a {
				t.Errorf("expected\n%s\ngot\n%s", e, a)
			}
			if result.ErrList != nil {
				if diff := cmp.Diff(tc.expectErrList, *result.ErrList, cmpOpts...); diff != "" {
					t.Errorf("unexpected field errors (-want,+got):\n%s", diff)
				}
			}
		})
	}
}
