/*
Copyright 2024 The Kubernetes Authors.

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

package schemawatcher

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/openapi"
)

func TestDetectChange(t *testing.T) {
	client := new(mockOpenAPIClient)
	watcher := New(client)
	for _, step := range []struct {
		paths    map[string]openapi.GroupVersion
		expected []schema.GroupVersionKind
	}{
		{
			// Initial step, empty
			paths:    map[string]openapi.GroupVersion{},
			expected: nil,
		},
		{
			// 1st step, apis.example.com/v1 presents, contains Foo
			paths: map[string]openapi.GroupVersion{
				"apis/apis.example.com/v1": &mockGroupVersion{json: jsonGVDocs[0], hash: "1"},
			},
			expected: []schema.GroupVersionKind{
				{
					Group:   "apis.example.com",
					Version: "v1",
					Kind:    "Foo",
				},
			},
		},
		{
			// 2nd step, apis.example.com/v1 mutates, contains Foo and Bar, Foo did not change
			paths: map[string]openapi.GroupVersion{
				"apis/apis.example.com/v1": &mockGroupVersion{json: jsonGVDocs[1], hash: "2"},
			},
			expected: []schema.GroupVersionKind{
				{
					Group:   "apis.example.com",
					Version: "v1",
					Kind:    "Bar",
				},
			},
		},
		{
			// 3rd step, apis.example.com/v1 mutates, contains Foo and Bar, Foo mutates
			paths: map[string]openapi.GroupVersion{
				"apis/apis.example.com/v1": &mockGroupVersion{json: jsonGVDocs[2], hash: "3"},
			},
			expected: []schema.GroupVersionKind{
				{
					Group:   "apis.example.com",
					Version: "v1",
					Kind:    "Foo",
				},
			},
		},
		{
			// 4th step, new version apis.example.com/v2, contains Foo
			// also v1 reverted to original
			paths: map[string]openapi.GroupVersion{
				"apis/apis.example.com/v1": &mockGroupVersion{json: jsonGVDocs[0], hash: "1"},
				"apis/apis.example.com/v2": &mockGroupVersion{json: strings.ReplaceAll(jsonGVDocs[0], "v1", "v2"), hash: "11"},
			},
			expected: []schema.GroupVersionKind{
				{
					Group:   "apis.example.com",
					Version: "v1",
					Kind:    "Foo", // reverted
				},
				{
					Group:   "apis.example.com",
					Version: "v1",
					Kind:    "Bar", // removed
				},
				{
					Group:   "apis.example.com",
					Version: "v2",
					Kind:    "Foo", // new
				},
			},
		},
		{
			// 5th step, apis.example.com/v1 unmounted
			paths: map[string]openapi.GroupVersion{
				"apis/apis.example.com/v2": &mockGroupVersion{json: strings.ReplaceAll(jsonGVDocs[0], "v1", "v2"), hash: "11"},
			},
			expected: []schema.GroupVersionKind{
				{
					Group:   "apis.example.com",
					Version: "v1",
					Kind:    "Foo", // removed
				},
			},
		},
	} {
		client.paths = step.paths
		changed, err := watcher.detectChanges()
		if err != nil {
			t.Fatalf("fail to detect changes: %v", err)
		}
		if !sets.New[schema.GroupVersionKind](step.expected...).Equal(sets.New[schema.GroupVersionKind](changed...)) {
			t.Fatalf("unexpected result, expected %v but got %v", step.expected, changed)
		}
	}
}

type mockOpenAPIClient struct {
	paths map[string]openapi.GroupVersion
}

type mockGroupVersion struct {
	json string
	hash string
}

var _ openapi.Client = (*mockOpenAPIClient)(nil)
var _ openapi.GroupVersion = (*mockGroupVersion)(nil)

func (c *mockOpenAPIClient) Paths() (map[string]openapi.GroupVersion, error) {
	return c.paths, nil
}

func (gv *mockGroupVersion) Schema(contentType string) ([]byte, error) {
	if contentType != runtime.ContentTypeJSON {
		panic("unexpected content type: " + contentType)
	}
	return []byte(gv.json), nil
}

func (gv *mockGroupVersion) Hash() (string, error) {
	return gv.hash, nil
}
