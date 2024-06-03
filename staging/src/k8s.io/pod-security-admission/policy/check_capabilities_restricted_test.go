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

func TestCapabilitiesRestricted_1_25(t *testing.T) {
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
					{Name: "a", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"FOO", "BAR"}}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"BAR", "BAZ"}}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"NET_BIND_SERVICE", "CHOWN"}, Drop: []corev1.Capability{"ALL", "FOO"}}}},
				}}},
			expectReason: `unrestricted capabilities`,
			expectDetail: `containers "a", "b" must set securityContext.capabilities.drop=["ALL"]; containers "a", "b", "c" must not include "BAR", "BAZ", "CHOWN", "FOO" in securityContext.capabilities.add`,
		},
		{
			name: "multiple containers, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"FOO", "BAR"}}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"BAR", "BAZ"}}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"NET_BIND_SERVICE", "CHOWN"}, Drop: []corev1.Capability{"ALL", "FOO"}}}},
				}}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `unrestricted capabilities`,
			expectDetail: `containers "a", "b" must set securityContext.capabilities.drop=["ALL"]; containers "a", "b", "c" must not include "BAR", "BAZ", "CHOWN", "FOO" in securityContext.capabilities.add`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[0].securityContext.capabilities.add", BadValue: []string{"BAR", "FOO"}},
				{Type: field.ErrorTypeRequired, Field: "spec.containers[0].securityContext.capabilities.drop", BadValue: ""},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[1].securityContext.capabilities.add", BadValue: []string{"BAR", "BAZ"}},
				{Type: field.ErrorTypeRequired, Field: "spec.containers[1].securityContext.capabilities.drop", BadValue: ""},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[2].securityContext.capabilities.add", BadValue: []string{"CHOWN"}},
			},
		},
		{
			name: "container is not allowed on both Add and Drop",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"BAR", "BAZ"}, Drop: []corev1.Capability{"FOO", "BAR"}}}},
				}}},
			expectReason: `unrestricted capabilities`,
			expectDetail: `container "a" must set securityContext.capabilities.drop=["ALL"]; container "a" must not include "BAR", "BAZ" in securityContext.capabilities.add`,
		},
		{
			name: "container is not allowed on both Add and Drop, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"BAR", "BAZ"}, Drop: []corev1.Capability{"FOO", "BAR"}}}},
				}}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `unrestricted capabilities`,
			expectDetail: `container "a" must set securityContext.capabilities.drop=["ALL"]; container "a" must not include "BAR", "BAZ" in securityContext.capabilities.add`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[0].securityContext.capabilities.add", BadValue: []string{"BAR", "BAZ"}},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[0].securityContext.capabilities.drop", BadValue: []string{"BAR", "FOO"}},
			},
		},
		{
			name: "windows pod, admit without checking capabilities",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				OS: &corev1.PodOS{Name: corev1.Windows},
				Containers: []corev1.Container{
					{Name: "a"},
				}}},
			allowed: true,
		},
		{
			name: "windows pod, admit without checking capabilities, enable field error list",
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
			expectReason: `unrestricted capabilities`,
			expectDetail: `container "a" must set securityContext.capabilities.drop=["ALL"]`,
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
			expectReason: `unrestricted capabilities`,
			expectDetail: `container "a" must set securityContext.capabilities.drop=["ALL"]`,
			allowed:      false,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "spec.containers[0].securityContext.capabilities.drop", BadValue: ""},
			},
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := capabilitiesRestrictedV1Dot25(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
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

func TestCapabilitiesRestricted_1_22(t *testing.T) {
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
					{Name: "a", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"FOO", "BAR"}}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"BAR", "BAZ"}}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"NET_BIND_SERVICE", "CHOWN"}, Drop: []corev1.Capability{"ALL", "FOO"}}}},
				}}},
			expectReason: `unrestricted capabilities`,
			expectDetail: `containers "a", "b" must set securityContext.capabilities.drop=["ALL"]; containers "a", "b", "c" must not include "BAR", "BAZ", "CHOWN", "FOO" in securityContext.capabilities.add`,
		},
		{
			name: "multiple containers enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"FOO", "BAR"}}}},
					{Name: "b", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"BAR", "BAZ"}}}},
					{Name: "c", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"NET_BIND_SERVICE", "CHOWN"}, Drop: []corev1.Capability{"ALL", "FOO"}}}},
				}}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `unrestricted capabilities`,
			expectDetail: `containers "a", "b" must set securityContext.capabilities.drop=["ALL"]; containers "a", "b", "c" must not include "BAR", "BAZ", "CHOWN", "FOO" in securityContext.capabilities.add`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[0].securityContext.capabilities.add", BadValue: []string{"BAR", "FOO"}},
				{Type: field.ErrorTypeRequired, Field: "spec.containers[0].securityContext.capabilities.drop", BadValue: ""},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[1].securityContext.capabilities.add", BadValue: []string{"BAR", "BAZ"}},
				{Type: field.ErrorTypeRequired, Field: "spec.containers[1].securityContext.capabilities.drop", BadValue: ""},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[2].securityContext.capabilities.add", BadValue: []string{"CHOWN"}},
			},
		},
		{
			name: "container is not allowed on both Add and Drop",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"BAR", "BAZ"}, Drop: []corev1.Capability{"FOO", "BAR"}}}},
				}}},
			expectReason: `unrestricted capabilities`,
			expectDetail: `container "a" must set securityContext.capabilities.drop=["ALL"]; container "a" must not include "BAR", "BAZ" in securityContext.capabilities.add`,
		},
		{
			name: "container is not allowed on both Add and Drop, enable field error list",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "a", SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"BAR", "BAZ"}, Drop: []corev1.Capability{"FOO", "BAR"}}}},
				}}},
			opts: options{
				withFieldErrors: true,
			},
			expectReason: `unrestricted capabilities`,
			expectDetail: `container "a" must set securityContext.capabilities.drop=["ALL"]; container "a" must not include "BAR", "BAZ" in securityContext.capabilities.add`,
			expectErrList: field.ErrorList{
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[0].securityContext.capabilities.add", BadValue: []string{"BAR", "BAZ"}},
				{Type: field.ErrorTypeForbidden, Field: "spec.containers[0].securityContext.capabilities.drop", BadValue: []string{"BAR", "FOO"}},
			},
		},
	}

	cmpOpts := []cmp.Option{cmpopts.IgnoreFields(field.Error{}, "Detail"), cmpopts.SortSlices(func(a, b *field.Error) bool { return a.Error() < b.Error() })}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := capabilitiesRestrictedV1Dot22(&tc.pod.ObjectMeta, &tc.pod.Spec, tc.opts)
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
