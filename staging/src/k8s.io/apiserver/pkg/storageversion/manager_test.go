/*
Copyright 2020 The Kubernetes Authors.

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

package storageversion

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestSortResourceInfosByGroupResource(t *testing.T) {
	tests := []struct {
		infos    []ResourceInfo
		expected []ResourceInfo
	}{
		{
			infos:    nil,
			expected: nil,
		},
		{
			infos:    []ResourceInfo{},
			expected: []ResourceInfo{},
		},
		{
			infos: []ResourceInfo{
				{GroupResource: schema.GroupResource{Group: "", Resource: "pods"}},
				{GroupResource: schema.GroupResource{Group: "", Resource: "nodes"}},
				{GroupResource: schema.GroupResource{Group: "networking.k8s.io", Resource: "ingresses"}},
				{GroupResource: schema.GroupResource{Group: "extensions", Resource: "ingresses"}},
			},
			expected: []ResourceInfo{
				{GroupResource: schema.GroupResource{Group: "", Resource: "nodes"}},
				{GroupResource: schema.GroupResource{Group: "", Resource: "pods"}},
				{GroupResource: schema.GroupResource{Group: "extensions", Resource: "ingresses"}},
				{GroupResource: schema.GroupResource{Group: "networking.k8s.io", Resource: "ingresses"}},
			},
		},
	}

	for _, tc := range tests {
		sortResourceInfosByGroupResource(tc.infos)
		if e, a := tc.expected, tc.infos; !reflect.DeepEqual(e, a) {
			t.Errorf("unexpected: %v", cmp.Diff(e, a))
		}
	}
}
