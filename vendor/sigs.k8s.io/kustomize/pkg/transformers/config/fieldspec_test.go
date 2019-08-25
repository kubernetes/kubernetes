/*
Copyright 2018 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/gvk"
)

func TestPathSlice(t *testing.T) {
	type path struct {
		input  string
		parsed []string
	}
	paths := []path{
		{
			input:  "spec/metadata/annotations",
			parsed: []string{"spec", "metadata", "annotations"},
		},
		{
			input:  `metadata/annotations/nginx.ingress.kubernetes.io\/auth-secret`,
			parsed: []string{"metadata", "annotations", "nginx.ingress.kubernetes.io/auth-secret"},
		},
	}
	for _, p := range paths {
		fs := FieldSpec{Path: p.input}
		actual := fs.PathSlice()
		if !reflect.DeepEqual(actual, p.parsed) {
			t.Fatalf("expected %v, but got %v", p.parsed, actual)
		}
	}
}

var mergeTests = []struct {
	name     string
	original fsSlice
	incoming fsSlice
	err      error
	result   fsSlice
}{
	{
		"normal",
		fsSlice{
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "apple"},
				CreateIfNotPresent: false,
			},
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "pear"},
				CreateIfNotPresent: false,
			},
		},
		fsSlice{
			{
				Path:               "home",
				Gvk:                gvk.Gvk{Group: "beans"},
				CreateIfNotPresent: false,
			},
		},
		nil,
		fsSlice{
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "apple"},
				CreateIfNotPresent: false,
			},
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "pear"},
				CreateIfNotPresent: false,
			},
			{
				Path:               "home",
				Gvk:                gvk.Gvk{Group: "beans"},
				CreateIfNotPresent: false,
			},
		},
	},
	{
		"ignore copy",
		fsSlice{
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "apple"},
				CreateIfNotPresent: false,
			},
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "pear"},
				CreateIfNotPresent: false,
			},
		},
		fsSlice{
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "apple"},
				CreateIfNotPresent: false,
			},
		},
		nil,
		fsSlice{
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "apple"},
				CreateIfNotPresent: false,
			},
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "pear"},
				CreateIfNotPresent: false,
			},
		},
	},
	{
		"error on conflict",
		fsSlice{
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "apple"},
				CreateIfNotPresent: false,
			},
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "pear"},
				CreateIfNotPresent: false,
			},
		},
		fsSlice{
			{
				Path:               "whatever",
				Gvk:                gvk.Gvk{Group: "apple"},
				CreateIfNotPresent: true,
			},
		},
		fmt.Errorf("hey"),
		fsSlice{},
	},
}

func TestFsSlice_MergeAll(t *testing.T) {
	for _, item := range mergeTests {
		result, err := item.original.mergeAll(item.incoming)
		if item.err == nil {
			if err != nil {
				t.Fatalf("test %s: unexpected err %v", item.name, err)
			}
			if !reflect.DeepEqual(item.result, result) {
				t.Fatalf("test %s: expected: %v\n but got: %v\n",
					item.name, item.result, result)
			}
		} else {
			if err == nil {
				t.Fatalf("test %s: expected err: %v", item.name, err)
			}
			if !strings.Contains(err.Error(), "conflicting fieldspecs") {
				t.Fatalf("test %s: unexpected err: %v", item.name, err)
			}
		}
	}
}
