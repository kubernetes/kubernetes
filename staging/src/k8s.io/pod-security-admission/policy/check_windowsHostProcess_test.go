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
	utilpointer "k8s.io/utils/pointer"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestWindowsHostProcess(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		opts          options
		expectReason  string
		expectDetail  string
		expectErrList field.ErrorList
	}{
		{
			name: "host process",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					WindowsOptions: &corev1.WindowsSecurityContextOptions{HostProcess: utilpointer.Bool(true)},
				},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{WindowsOptions: &corev1.WindowsSecurityContextOptions{}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{WindowsOptions: &corev1.WindowsSecurityContextOptions{HostProcess: utilpointer.Bool(false)}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{WindowsOptions: &corev1.WindowsSecurityContextOptions{HostProcess: utilpointer.Bool(true)}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{WindowsOptions: &corev1.WindowsSecurityContextOptions{HostProcess: utilpointer.Bool(true)}}},
				},
			}},
			expectReason: `hostProcess`,
			expectDetail: `pod and containers "e", "f" must not set securityContext.windowsOptions.hostProcess=true`,
		},
		{
			name: "host process, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{
					WindowsOptions: &corev1.WindowsSecurityContextOptions{HostProcess: utilpointer.Bool(true)},
				},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{WindowsOptions: &corev1.WindowsSecurityContextOptions{}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{WindowsOptions: &corev1.WindowsSecurityContextOptions{HostProcess: utilpointer.Bool(false)}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{WindowsOptions: &corev1.WindowsSecurityContextOptions{HostProcess: utilpointer.Bool(true)}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{WindowsOptions: &corev1.WindowsSecurityContextOptions{HostProcess: utilpointer.Bool(true)}}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `hostProcess`,
			expectDetail: `pod and containers "e", "f" must not set securityContext.windowsOptions.hostProcess=true`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.windowsOptions.hostProcess", BadValue: true},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[4].securityContext.windowsOptions.hostProcess", BadValue: true},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[5].securityContext.windowsOptions.hostProcess", BadValue: true},
			},
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := windowsHostProcessV1Dot0(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
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
