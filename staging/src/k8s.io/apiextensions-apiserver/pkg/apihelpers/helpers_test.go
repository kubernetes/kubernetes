/*
Copyright 2019 The Kubernetes Authors.

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

package apihelpers

import (
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
)

func TestIsProtectedCommunityGroup(t *testing.T) {
	tests := []struct {
		name string

		group    string
		expected bool
	}{
		{
			name:     "bare k8s",
			group:    "k8s.io",
			expected: true,
		},
		{
			name:     "bare kube",
			group:    "kubernetes.io",
			expected: true,
		},
		{
			name:     "nested k8s",
			group:    "sigs.k8s.io",
			expected: true,
		},
		{
			name:     "nested kube",
			group:    "sigs.kubernetes.io",
			expected: true,
		},
		{
			name:     "alternative",
			group:    "different.io",
			expected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := IsProtectedCommunityGroup(test.group)

			if actual != test.expected {
				t.Fatalf("expected %v, got %v", test.expected, actual)
			}
		})
	}
}

func TestGetAPIApprovalState(t *testing.T) {
	tests := []struct {
		name string

		annotations map[string]string
		expected    APIApprovalState
	}{
		{
			name:        "bare unapproved",
			annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "unapproved"},
			expected:    APIApprovalBypassed,
		},
		{
			name:        "unapproved with message",
			annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "unapproved, experimental-only"},
			expected:    APIApprovalBypassed,
		},
		{
			name:        "mismatched case",
			annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "Unapproved"},
			expected:    APIApprovalInvalid,
		},
		{
			name:        "empty",
			annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: ""},
			expected:    APIApprovalMissing,
		},
		{
			name:        "missing",
			annotations: map[string]string{},
			expected:    APIApprovalMissing,
		},
		{
			name:        "url",
			annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "https://github.com/kubernetes/kubernetes/pull/78458"},
			expected:    APIApproved,
		},
		{
			name:        "url - no scheme",
			annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "github.com/kubernetes/kubernetes/pull/78458"},
			expected:    APIApprovalInvalid,
		},
		{
			name:        "url - no host",
			annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "http:///kubernetes/kubernetes/pull/78458"},
			expected:    APIApprovalInvalid,
		},
		{
			name:        "url - just path",
			annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "/"},
			expected:    APIApprovalInvalid,
		},
		{
			name:        "missing scheme",
			annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: "github.com/kubernetes/kubernetes/pull/78458"},
			expected:    APIApprovalInvalid,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual, _ := GetAPIApprovalState(test.annotations)

			if actual != test.expected {
				t.Fatalf("expected %v, got %v", test.expected, actual)
			}
		})
	}
}
