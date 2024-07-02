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

func TestAllowPrivilegeEscalation_1_25(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		opts          options
		expectReason  string
		expectDetail  string
		allowed       bool
		expectErrList field.ErrorList
	}{
		{
			name: "multiple containers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
					{Name: "b", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: nil}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(true)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(false)}},
				}}},
			expectReason: `allowPrivilegeEscalation != false`,
			expectDetail: `containers "a", "b", "c" must set securityContext.allowPrivilegeEscalation=false`,
			allowed:      false,
		},
		{
			name: "multiple containers, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
					{Name: "b", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: nil}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(true)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(false)}},
				}}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `allowPrivilegeEscalation != false`,
			expectDetail: `containers "a", "b", "c" must set securityContext.allowPrivilegeEscalation=false`,
			allowed:      false,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "spec.containers[0].securityContext.allowPrivilegeEscalation", BadValue: ""},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[1].securityContext.allowPrivilegeEscalation", BadValue: "nil"},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[2].securityContext.allowPrivilegeEscalation", BadValue: true},
			},
		},
		{
			name: "windows pod, admit without checking privilegeEscalation",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				OS: &corev1.PodOS{Name: corev1.Windows},
				Containers: []corev1.Container{
					{Name: "a"},
				}}},
			allowed: true,
		},
		{
			name: "windows pod, admit without checking privilegeEscalation, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				OS: &corev1.PodOS{Name: corev1.Windows},
				Containers: []corev1.Container{
					{Name: "a"},
				}}},
			opts: options{
				withFieldErrors: true,
			},
			allowed: true,
		},
		{
			name: "linux pod, reject if security context is not set",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				OS: &corev1.PodOS{Name: corev1.Linux},
				Containers: []corev1.Container{
					{Name: "a"},
				}}},
			expectReason: `allowPrivilegeEscalation != false`,
			expectDetail: `container "a" must set securityContext.allowPrivilegeEscalation=false`,
			allowed:      false,
		},
		{
			name: "linux pod, reject if security context is not set, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				OS: &corev1.PodOS{Name: corev1.Linux},
				Containers: []corev1.Container{
					{Name: "a"},
				}}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `allowPrivilegeEscalation != false`,
			expectDetail: `container "a" must set securityContext.allowPrivilegeEscalation=false`,
			allowed:      false,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "spec.containers[0].securityContext.allowPrivilegeEscalation", BadValue: ""},
			},
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := allowPrivilegeEscalationV1Dot25(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
			if result.Allowed && !tc.allowed {
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

func TestAllowPrivilegeEscalation_1_8(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		opts          options
		expectReason  string
		expectDetail  string
		expectErrList field.ErrorList
	}{
		{
			name: "multiple containers",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
					{Name: "b", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: nil}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(true)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(false)}},
				}}},
			expectReason: `allowPrivilegeEscalation != false`,
			expectDetail: `containers "a", "b", "c" must set securityContext.allowPrivilegeEscalation=false`,
		},
		{
			name: "multiple containers enable field error List",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
					{Name: "b", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: nil}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(true)}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{AllowPrivilegeEscalation: utilpointer.Bool(false)}},
				}}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `allowPrivilegeEscalation != false`,
			expectDetail: `containers "a", "b", "c" must set securityContext.allowPrivilegeEscalation=false`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "spec.containers[0].securityContext.allowPrivilegeEscalation", BadValue: ""},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[1].securityContext.allowPrivilegeEscalation", BadValue: "nil"},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[2].securityContext.allowPrivilegeEscalation", BadValue: true},
			},
		},
	}

	ignoreDetail := cmpopts.IgnoreFields(field.Error{}, "Detail")
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := allowPrivilegeEscalationV1Dot8(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
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
				if diff := cmp.Diff(tc.expectErrList, *result.ErrList, ignoreDetail); diff != "" {
					t.Errorf("unexpected field errors (-want,+got):\n%s", diff)
				}
			}
		})
	}
}
