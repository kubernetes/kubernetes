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

package types

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
	"sigs.k8s.io/kustomize/pkg/gvk"
)

func TestGVK(t *testing.T) {
	type testcase struct {
		data     string
		expected gvk.Gvk
	}

	testcases := []testcase{
		{
			data: `
apiVersion: v1
kind: Secret
name: my-secret
`,
			expected: gvk.Gvk{Group: "", Version: "v1", Kind: "Secret"},
		},
		{
			data: `
apiVersion: myapps/v1
kind: MyKind
name: my-kind
`,
			expected: gvk.Gvk{Group: "myapps", Version: "v1", Kind: "MyKind"},
		},
		{
			data: `
version: v2
kind: MyKind
name: my-kind
`,
			expected: gvk.Gvk{Version: "v2", Kind: "MyKind"},
		},
	}

	for _, tc := range testcases {
		var targ Target
		err := yaml.Unmarshal([]byte(tc.data), &targ)
		if err != nil {
			t.Fatalf("Unexpected error %v", err)
		}
		if !reflect.DeepEqual(targ.GVK(), tc.expected) {
			t.Fatalf("Expected %v, but got %v", tc.expected, targ.GVK())
		}
	}
}

func TestDefaulting(t *testing.T) {
	v := &Var{
		Name: "SOME_VARIABLE_NAME",
		ObjRef: Target{
			Gvk: gvk.Gvk{
				Version: "v1",
				Kind:    "Secret",
			},
			Name: "my-secret",
		},
	}
	v.defaulting()
	if v.FieldRef.FieldPath != defaultFieldPath {
		t.Fatalf("expected %s, got %v",
			defaultFieldPath, v.FieldRef.FieldPath)
	}
}

func TestVarSet(t *testing.T) {
	set := VarSet{}
	vars := []Var{
		{
			Name: "SHELLVARS",
			ObjRef: Target{
				APIVersion: "v7",
				Gvk:        gvk.Gvk{Kind: "ConfigMap"},
				Name:       "bash"},
		},
		{
			Name: "BACKEND",
			ObjRef: Target{
				APIVersion: "v7",
				Gvk:        gvk.Gvk{Kind: "Deployment"},
				Name:       "myTiredBackend"},
		},
		{
			Name: "AWARD",
			ObjRef: Target{
				APIVersion: "v7",
				Gvk:        gvk.Gvk{Kind: "Service"},
				Name:       "nobelPrize"},
			FieldRef: FieldSelector{FieldPath: "some.arbitrary.path"},
		},
	}
	err := set.MergeSlice(vars)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	for _, v := range vars {
		if !set.Contains(v) {
			t.Fatalf("set %v should contain var %v", set.AsSlice(), v)
		}
	}
	set2 := VarSet{}
	err = set2.MergeSet(set)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	err = set2.MergeSlice(vars)
	if err == nil {
		t.Fatalf("expected err")
	}
	if !strings.Contains(err.Error(), "var 'SHELLVARS' already encountered") {
		t.Fatalf("unexpected err: %v", err)
	}
	v := set2.Get("BACKEND")
	if v == nil {
		t.Fatalf("expected var")
	}
	// Confirm defaulting.
	if v.FieldRef.FieldPath != defaultFieldPath {
		t.Fatalf("unexpected field path: %v", v.FieldRef.FieldPath)
	}
	// Confirm sorting.
	names := set2.AsSlice()
	if names[0].Name != "AWARD" ||
		names[1].Name != "BACKEND" ||
		names[2].Name != "SHELLVARS" {
		t.Fatalf("unexpected order in : %v", names)
	}
}
