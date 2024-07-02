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

func TestSeccompProfileRestricted_1_25(t *testing.T) {
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
			name: "no explicit seccomp",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `pod or container "a" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
		},
		{
			name: "no explicit seccomp, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `pod or container "a" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "spec.containers[0].securityContext.seccompProfile.type", BadValue: ""},
			},
		},
		{
			name: "no explicit seccomp, windows Pod",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				OS: &corev1.PodOS{Name: corev1.Windows},
				Containers: []corev1.Container{
					{Name: "a"},
				},
			}},
			allowed: true,
		},
		{
			name: "no explicit seccomp, windows Pod, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				OS: &corev1.PodOS{Name: corev1.Windows},
				Containers: []corev1.Container{
					{Name: "a"},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			allowed: true,
		},
		{
			name: "no explicit seccomp, linux pod",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				OS: &corev1.PodOS{Name: corev1.Linux},
				Containers: []corev1.Container{
					{Name: "a"},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `pod or container "a" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
			allowed:      false,
		},
		{
			name: "no explicit seccomp, linux pod, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				OS: &corev1.PodOS{Name: corev1.Linux},
				Containers: []corev1.Container{
					{Name: "a"},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `pod or container "a" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
			allowed:      false,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "spec.containers[0].securityContext.seccompProfile.type", BadValue: ""},
			},
		},
		{
			name: "pod seccomp invalid",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `pod must not set securityContext.seccompProfile.type to "Unconfined"`,
		},
		{
			name: "pod seccomp invalid, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `pod must not set securityContext.seccompProfile.type to "Unconfined"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seccompProfile.type", BadValue: "Unconfined"},
			},
		},
		{
			name: "containers seccomp invalid",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `containers "c", "d" must not set securityContext.seccompProfile.type to "Unconfined"`,
		},
		{
			name: "containers seccomp invalid, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `containers "c", "d" must not set securityContext.seccompProfile.type to "Unconfined"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[2].securityContext.seccompProfile.type", BadValue: "Unconfined"},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[3].securityContext.seccompProfile.type", BadValue: "Unconfined"},
			},
		},
		{
			name: "pod nil, container fallthrough",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `pod or containers "a", "b" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
		},
		{
			name: "pod nil, container fallthrough, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `pod or containers "a", "b" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "spec.containers[0].securityContext.seccompProfile.type", BadValue: ""},
				{Type: field.ErrorTypeRequired, Field: "spec.containers[1].securityContext.seccompProfile.type", BadValue: ""},
			},
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := seccompProfileRestrictedV1Dot25(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
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

func TestSeccompProfileRestricted_1_19(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		opts          options
		expectReason  string
		expectDetail  string
		expectErrList field.ErrorList
	}{
		{
			name: "no explicit seccomp",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `pod or container "a" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
		},
		{
			name: "no explicit seccomp, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a"},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `pod or container "a" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "spec.containers[0].securityContext.seccompProfile.type", BadValue: ""},
			},
		},
		{
			name: "pod seccomp invalid",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `pod must not set securityContext.seccompProfile.type to "Unconfined"`,
		},
		{
			name: "pod seccomp invalid, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `pod must not set securityContext.seccompProfile.type to "Unconfined"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seccompProfile.type", BadValue: "Unconfined"},
			},
		},
		{
			name: "containers seccomp invalid",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `containers "c", "d" must not set securityContext.seccompProfile.type to "Unconfined"`,
		},
		{
			name: "containers seccomp invalid, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}},
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
					{Name: "f", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `containers "c", "d" must not set securityContext.seccompProfile.type to "Unconfined"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[2].securityContext.seccompProfile.type", BadValue: "Unconfined"},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[3].securityContext.seccompProfile.type", BadValue: "Unconfined"},
			},
		},
		{
			name: "pod nil, container fallthrough",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
				},
			}},
			expectReason: `seccompProfile`,
			expectDetail: `pod or containers "a", "b" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
		},
		{
			name: "pod nil, container fallthrough, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: nil},
					{Name: "b", SecurityContext: &corev1.SecurityContext{}},
					{Name: "d", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
					{Name: "e", SecurityContext: &corev1.SecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeRuntimeDefault}}},
				},
			}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `pod or containers "a", "b" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "spec.containers[0].securityContext.seccompProfile.type", BadValue: ""},
				{Type: field.ErrorTypeRequired, Field: "spec.containers[1].securityContext.seccompProfile.type", BadValue: ""},
			},
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := seccompProfileRestrictedV1Dot19(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
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
