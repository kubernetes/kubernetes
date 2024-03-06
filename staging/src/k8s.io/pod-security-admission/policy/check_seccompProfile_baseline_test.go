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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestSeccompProfileBaseline_1_0(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		opts          options
		expectReason  string
		expectDetail  string
		expectErrList field.ErrorList
	}{
		{
			name: "pod seccomp invalid",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"seccomp.security.alpha.kubernetes.io/pod": "unconfined",
					},
				},
			},
			expectReason: `seccompProfile`,
			expectDetail: `forbidden annotation seccomp.security.alpha.kubernetes.io/pod="unconfined"`,
		},
		{
			name: "pod seccomp invalid, enable field error list",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"seccomp.security.alpha.kubernetes.io/pod": "unconfined",
					},
				},
			},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `forbidden annotation seccomp.security.alpha.kubernetes.io/pod="unconfined"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "metadata.annotations[seccomp.security.alpha.kubernetes.io/pod]", BadValue: "unconfined"},
			},
		},
		{
			name: "containers seccomp invalid",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"container.seccomp.security.alpha.kubernetes.io/a": "unconfined",
						"container.seccomp.security.alpha.kubernetes.io/b": "unknown",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "a"}, {Name: "b"}, {Name: "c"}},
				},
			},
			expectReason: `seccompProfile`,
			expectDetail: `forbidden annotations container.seccomp.security.alpha.kubernetes.io/a="unconfined", container.seccomp.security.alpha.kubernetes.io/b="unknown"`,
		},
		{
			name: "containers seccomp invalid, enable field error list",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"container.seccomp.security.alpha.kubernetes.io/a": "unconfined",
						"container.seccomp.security.alpha.kubernetes.io/b": "unknown",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "a"}, {Name: "b"}, {Name: "c"}},
				},
			},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `forbidden annotations container.seccomp.security.alpha.kubernetes.io/a="unconfined", container.seccomp.security.alpha.kubernetes.io/b="unknown"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "metadata.annotations[container.seccomp.security.alpha.kubernetes.io/a]", BadValue: "unconfined"},
				{Type: field.ErrorTypeForbidden, Field: "metadata.annotations[container.seccomp.security.alpha.kubernetes.io/b]", BadValue: "unknown"},
			},
		},
		{
			name: "pod and containers seccomp invalid",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"seccomp.security.alpha.kubernetes.io/pod":         "unconfined",
						"container.seccomp.security.alpha.kubernetes.io/a": "unconfined",
						"container.seccomp.security.alpha.kubernetes.io/b": "unknown",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "a"}, {Name: "b"}, {Name: "c"}},
				},
			},
			expectReason: `seccompProfile`,
			expectDetail: `forbidden annotations container.seccomp.security.alpha.kubernetes.io/a="unconfined", container.seccomp.security.alpha.kubernetes.io/b="unknown", seccomp.security.alpha.kubernetes.io/pod="unconfined"`,
		},
		{
			name: "pod and containers seccomp invalid, enable field error list",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"seccomp.security.alpha.kubernetes.io/pod":         "unconfined",
						"container.seccomp.security.alpha.kubernetes.io/a": "unconfined",
						"container.seccomp.security.alpha.kubernetes.io/b": "unknown",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "a"}, {Name: "b"}, {Name: "c"}},
				},
			},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `seccompProfile`,
			expectDetail: `forbidden annotations container.seccomp.security.alpha.kubernetes.io/a="unconfined", container.seccomp.security.alpha.kubernetes.io/b="unknown", seccomp.security.alpha.kubernetes.io/pod="unconfined"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "metadata.annotations[seccomp.security.alpha.kubernetes.io/pod]", BadValue: "unconfined"},
				{Type: field.ErrorTypeForbidden, Field: "metadata.annotations[container.seccomp.security.alpha.kubernetes.io/a]", BadValue: "unconfined"},
				{Type: field.ErrorTypeForbidden, Field: "metadata.annotations[container.seccomp.security.alpha.kubernetes.io/b]", BadValue: "unknown"},
			},
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := seccompProfileBaselineV1Dot0(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
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

func TestSeccompProfileBaseline_1_19(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		opts          options
		expectReason  string
		expectDetail  string
		expectErrList field.ErrorList
	}{
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
			name: "pod and containers seccomp invalid",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}},
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
			expectDetail: `pod and containers "c", "d" must not set securityContext.seccompProfile.type to "Unconfined"`,
		},
		{
			name: "pod and containers seccomp invalid, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				SecurityContext: &corev1.PodSecurityContext{SeccompProfile: &corev1.SeccompProfile{Type: corev1.SeccompProfileTypeUnconfined}},
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
			expectDetail: `pod and containers "c", "d" must not set securityContext.seccompProfile.type to "Unconfined"`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.securityContext.seccompProfile.type", BadValue: "Unconfined"},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[2].securityContext.seccompProfile.type", BadValue: "Unconfined"},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[3].securityContext.seccompProfile.type", BadValue: "Unconfined"},
			},
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := seccompProfileBaselineV1Dot19(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
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
