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

func TestRunAsUser(t *testing.T) {
	tests := []struct {
		name                                     string
		pod                                      *corev1.Pod
		opts                                     options
		expectAllow                              bool
		expectReason                             string
		expectDetail                             string
		expectErrList                            field.ErrorList
		enableUserNamespacesPodSecurityStandards bool
	}{
		{
			name: "pod runAsUser=0",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: utilpointer.Int64(0)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			expectReason: `runAsUser=0`,
			expectDetail: `pod must not set runAsUser=0`,
		},
		{
			name: "pod runAsUser=0, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: utilpointer.Int64(0)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `runAsUser=0`,
			expectDetail: `pod must not set runAsUser=0`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.runAsUser", BadValue: 0},
			},
		},
		{
			name: "pod runAsUser=non-zero",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: utilpointer.Int64(1000)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			expectAllow: true,
		},
		{
			name: "pod runAsUser=non-zero, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: utilpointer.Int64(1000)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectAllow: true,
		},
		{
			name: "pod runAsUser=nil",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: nil},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			expectAllow: true,
		},
		{
			name: "pod runAsUser=nil, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: nil},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectAllow: true,
		},
		{
			name: "containers runAsUser=0",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: utilpointer.Int64(1000)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(0)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(0)}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(1)}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(1)}},
				},
			}},
			expectReason: `runAsUser=0`,
			expectDetail: `containers "c", "d" must not set runAsUser=0`,
		},
		{
			name: "containers runAsUser=0, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: utilpointer.Int64(1000)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(0)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(0)}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(1)}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(1)}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `runAsUser=0`,
			expectDetail: `containers "c", "d" must not set runAsUser=0`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[2].securityContext.runAsUser", BadValue: 0},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[3].securityContext.runAsUser", BadValue: 0},
			},
		},
		{
			name: "containers runAsUser=non-zero",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "c", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(1)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(2)}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(3)}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(4)}},
				},
			}},
			expectAllow: true,
		},
		{
			name: "containers runAsUser=non-zero, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "c", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(1)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(2)}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(3)}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{RunAsUser: utilpointer.Int64(4)}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectAllow: true,
		},
		{
			name: "UserNamespacesPodSecurityStandards enabled without HostUsers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				HostUsers: utilpointer.Bool(false),
			}},
			expectAllow:                              true,
			enableUserNamespacesPodSecurityStandards: true,
		},
		{
			name: "UserNamespacesPodSecurityStandards enabled without HostUsers, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				HostUsers: utilpointer.Bool(false),
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectAllow:                              true,
			enableUserNamespacesPodSecurityStandards: true,
		},
		{
			name: "UserNamespacesPodSecurityStandards enabled with HostUsers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: utilpointer.Int64(0)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
				HostUsers: utilpointer.Bool(true),
			}},
			expectReason:                             `runAsUser=0`,
			expectDetail:                             `pod must not set runAsUser=0`,
			enableUserNamespacesPodSecurityStandards: true,
		},
		{
			name: "UserNamespacesPodSecurityStandards enabled with HostUsers, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{RunAsUser: utilpointer.Int64(0)},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
				HostUsers: utilpointer.Bool(true),
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `runAsUser=0`,
			expectDetail: `pod must not set runAsUser=0`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.runAsUser", BadValue: 0},
			},
			enableUserNamespacesPodSecurityStandards: true,
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.enableUserNamespacesPodSecurityStandards {
				RelaxPolicyForUserNamespacePods(true)
			}
			result := runAsUserV1Dot23(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
			if tc.expectAllow {
				if !result.Allowed {
					t.Fatalf("expected to be allowed, disallowed: %s, %s", result.ForbiddenReason, result.ForbiddenDetail)
				}
				return
			}
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
