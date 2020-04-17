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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestNewStategicMergeSliceFromBytes(t *testing.T) {
	var useCases = []struct {
		name            string
		in              string
		expectedPatches int
		expectedError   bool
	}{
		{
			name:            "empty",
			in:              "",
			expectedPatches: 0,
		},
		{
			name: "single patch",
			in: dedent.Dedent(`
            apiVersion: v1
            kind: Pod
            metadata:
              name: kube-apiserver
            `),
			expectedPatches: 1,
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
			expectedPatches: 2,
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
			expectedPatches: 2,
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
			expectedPatches: 2,
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
			r, err := newStrategicMergeSliceFromBytes([]byte(rt.in))
			if err != nil {
				if !rt.expectedError {
					t.Errorf("newStrategicMergeSliceFromBytes returned unexpected error: %v", err)
				}
				return
			}
			if err == nil && rt.expectedError {
				t.Error("newStrategicMergeSliceFromBytes does not returned expected error")
			}
			if len(r) != rt.expectedPatches {
				t.Errorf("Expected %d strategic merge patches in the slice, actual %d", rt.expectedPatches, len(r))
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
	u, err := newStrategicMergeSliceFromBytes([]byte(in))
	if err != nil {
		t.Fatalf("newStategicMergeSliceFromBytes returned unexpected error: %v", err)
	}

	var useCases = []struct {
		name            string
		rgvk            schema.GroupVersionKind
		rnamespace      string
		rname           string
		expectedPatches int
	}{
		{
			name:            "match 1",
			rgvk:            schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
			rnamespace:      "kube-system",
			rname:           "kube-apiserver",
			expectedPatches: 1,
		},
		{
			name:            "match 2",
			rgvk:            schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
			rnamespace:      "kube-system",
			rname:           "kube-scheduler",
			expectedPatches: 2,
		},
		{
			name:            "match 0 (wrong gvk)",
			rgvk:            schema.GroupVersionKind{Group: "something", Version: "v1", Kind: "Pod"},
			rnamespace:      "kube-system",
			rname:           "kube-scheduler",
			expectedPatches: 0,
		},
		{
			name:            "match 0 (wrong namespace)",
			rgvk:            schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
			rnamespace:      "kube-something",
			rname:           "kube-scheduler",
			expectedPatches: 0,
		},
		{
			name:            "match 0 (wrong namr)",
			rgvk:            schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
			rnamespace:      "kube-system",
			rname:           "kube-something",
			expectedPatches: 0,
		},
	}
	for _, rt := range useCases {
		t.Run(rt.name, func(t *testing.T) {
			resource := &unstructured.Unstructured{}
			resource.SetGroupVersionKind(rt.rgvk)
			resource.SetNamespace(rt.rnamespace)
			resource.SetName(rt.rname)

			r := u.filterByResource(resource)

			if len(r) != rt.expectedPatches {
				t.Errorf("Expected %d strategic merge patches in the slice, actual %d", rt.expectedPatches, len(r))
			}
		})
	}
}
