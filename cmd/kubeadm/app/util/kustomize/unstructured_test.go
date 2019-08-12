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

package kustomize

import (
	"testing"

	"github.com/lithammer/dedent"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestNewUnstructuredSliceFromBytes(t *testing.T) {
	var useCases = []struct {
		name                 string
		in                   string
		expectedUnctructured int
		expectedError        bool
	}{
		{
			name:                 "empty",
			in:                   "",
			expectedUnctructured: 0,
		},
		{
			name: "single patch",
			in: dedent.Dedent(`
            apiVersion: v1
            kind: Pod
            metadata:
              name: kube-apiserver
            `),
			expectedUnctructured: 1,
		},
		{
			name: "two patches as separated yaml documents",
			in: dedent.Dedent(`
            apiVersion: v1
            kind: Pod
            metadata:
              name: kube-apiserver
            ---
            apiVersion: v1
            kind: Pod
            metadata:
              name: kube-apiserver
            `),
			expectedUnctructured: 2,
		},
		{
			name: "two patches as a k8s list",
			in: dedent.Dedent(`
            apiVersion: v1
            kind: List
            items:
            - apiVersion: v1
              kind: Pod
              metadata:
                name: kube-apiserver
            - apiVersion: v1
              kind: Pod
              metadata:
                name: kube-apiserver
            `),
			expectedUnctructured: 2,
		},
		{
			name: "nested k8s lists",
			in: dedent.Dedent(`
            apiVersion: v1
            kind: List
            items:
            - apiVersion: v1
              kind: Pod
              metadata:
                name: kube-apiserver
            - apiVersion: v1
              kind: List
              items:
              - apiVersion: v1
                kind: Pod
                metadata:
                  name: kube-apiserver
            `),
			expectedUnctructured: 2,
		},
		{
			name:          "invalid yaml",
			in:            "$$$",
			expectedError: true,
		},
		{
			name: "invalid patch (missing kind)",
			in: dedent.Dedent(`
            apiVersion: v1
            #kind: Pod
            metadata:
              name: kube-apiserver
            `),
			expectedError: true,
		},
		{
			name: "invalid patch (missing name)",
			in: dedent.Dedent(`
            apiVersion: v1
            kind: Pod
            metadata:
              #name: kube-apiserver
            `),
			expectedError: true,
		},
	}
	for _, rt := range useCases {
		t.Run(rt.name, func(t *testing.T) {
			r, err := NewUnstructuredSliceFromBytes([]byte(rt.in))
			if err != nil {
				if !rt.expectedError {
					t.Errorf("NewUnstructuredSliceFromBytes returned unexpected error: %v", err)
				}
				return
			}
			if err == nil && rt.expectedError {
				t.Error("NewUnstructuredSliceFromBytes does not returned expected error")
			}
			if len(r) != rt.expectedUnctructured {
				t.Errorf("Expected %d Unstructured items in the slice, actual %d", rt.expectedUnctructured, len(r))
			}
		})
	}
}

func TestFilterResource(t *testing.T) {
	in := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
      name: kube-apiserver
      namespace: kube-system
    ---
    apiVersion: v1
    kind: Pod
    metadata:
      name: kube-scheduler
      namespace: kube-system
    ---
    apiVersion: v1
    kind: Pod
    metadata:
      name: kube-scheduler
      namespace: kube-system
    `)
	u, err := NewUnstructuredSliceFromBytes([]byte(in))
	if err != nil {
		t.Fatalf("NewUnstructuredSliceFromBytes returned unexpected error: %v", err)
	}

	var useCases = []struct {
		name                 string
		rgvk                 schema.GroupVersionKind
		rnamespace           string
		rname                string
		expectedUnctructured int
	}{
		{
			name:                 "match 1",
			rgvk:                 schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
			rnamespace:           "kube-system",
			rname:                "kube-apiserver",
			expectedUnctructured: 1,
		},
		{
			name:                 "match 2",
			rgvk:                 schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
			rnamespace:           "kube-system",
			rname:                "kube-scheduler",
			expectedUnctructured: 2,
		},
		{
			name:                 "match 0 (wrong gvk)",
			rgvk:                 schema.GroupVersionKind{Group: "something", Version: "v1", Kind: "Pod"},
			rnamespace:           "kube-system",
			rname:                "kube-scheduler",
			expectedUnctructured: 0,
		},
		{
			name:                 "match 0 (wrong namespace)",
			rgvk:                 schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
			rnamespace:           "kube-something",
			rname:                "kube-scheduler",
			expectedUnctructured: 0,
		},
		{
			name:                 "match 0 (wrong namr)",
			rgvk:                 schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
			rnamespace:           "kube-system",
			rname:                "kube-something",
			expectedUnctructured: 0,
		},
	}
	for _, rt := range useCases {
		t.Run(rt.name, func(t *testing.T) {
			r := u.FilterResource(rt.rgvk, rt.rnamespace, rt.rname)

			if len(r) != rt.expectedUnctructured {
				t.Errorf("Expected %d Unstructured items in the slice, actual %d", rt.expectedUnctructured, len(r))
			}
		})
	}
}
