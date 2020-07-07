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

package kubernetes_test

import (
	"io/ioutil"
	"testing"

	. "k8s.io/apimachinery/pkg/util/manifest"
)

func TestLoadAllItemsIntoFlattendList(t *testing.T) {
	for _, sample := range []struct {
		path        string
		expectedLen int
	}{
		{
			path:        "testdata/misc-sample-nested-list-1.json",
			expectedLen: 6,
		},
		{
			path:        "testdata/misc-sample-multidoc-nested-lists-1.yaml",
			expectedLen: 4,
		},
		{
			path:        "testdata/misc-sample-empty-list-1.json",
			expectedLen: 0,
		},
		{
			path:        "testdata/misc-sample-multidoc-empty-lists-1.yaml",
			expectedLen: 0,
		},
		{
			path:        "testdata/misc-sample-multidoc-empty-lists-2.yaml",
			expectedLen: 0,
		},
	} {
		data, err := ioutil.ReadFile(sample.path)
		if err != nil {
			t.Fatalf("unexpected error reading sample %q: %v", sample.path, err)
		}
		list, err := NewUnstructuredList(data)
		if err != nil {
			t.Fatalf("unexpected error converting data from %q into a list: %v", sample.path, err)
		}
		if list == nil {
			t.Fatalf("unexpected empty list from %q", sample.path)
		}
		list, err = Flatten(list)
		if err != nil {
			t.Fatalf("unexpected error flattening list parsed from %q: %v", sample.path, err)
		}
		if l := len(list.Items); l != sample.expectedLen {
			t.Fatalf("unexpected length of flattened list from %q: expected %d, got %d", sample.path, sample.expectedLen, l)
		}
	}
}
