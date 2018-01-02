/*
Copyright 2016 The Kubernetes Authors.

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

package generators

import (
	"reflect"
	"testing"

	"k8s.io/gengo/types"
)

func Test_isRootedUnder(t *testing.T) {
	testCases := []struct {
		path   string
		roots  []string
		expect bool
	}{
		{
			path:   "/foo/bar",
			roots:  nil,
			expect: false,
		},
		{
			path:   "/foo/bar",
			roots:  []string{},
			expect: false,
		},
		{
			path: "/foo/bar",
			roots: []string{
				"/bad",
			},
			expect: false,
		},
		{
			path: "/foo/bar",
			roots: []string{
				"/foo",
			},
			expect: true,
		},
		{
			path: "/foo/bar",
			roots: []string{
				"/bad",
				"/foo",
			},
			expect: true,
		},
		{
			path: "/foo/bar/qux/zorb",
			roots: []string{
				"/foo/bar/qux",
			},
			expect: true,
		},
		{
			path: "/foo/bar",
			roots: []string{
				"/foo/bar",
			},
			expect: true,
		},
		{
			path: "/foo/barn",
			roots: []string{
				"/foo/bar",
			},
			expect: false,
		},
		{
			path: "/foo/bar",
			roots: []string{
				"/foo/barn",
			},
			expect: false,
		},
		{
			path: "/foo/bar",
			roots: []string{
				"",
			},
			expect: true,
		},
	}

	for i, tc := range testCases {
		r := isRootedUnder(tc.path, tc.roots)
		if r != tc.expect {
			t.Errorf("case[%d]: expected %t, got %t for %q in %q", i, tc.expect, r, tc.path, tc.roots)
		}
	}
}

func Test_hasDeepCopyMethod(t *testing.T) {
	testCases := []struct {
		typ    types.Type
		expect bool
	}{
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				// No DeepCopy method.
				Methods: map[string]*types.Type{},
			},
			expect: false,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// No DeepCopy method.
					"method": {
						Name: types.Name{Package: "pkgname", Name: "func()"},
						Kind: types.Func,
						Signature: &types.Signature{
							Parameters: []*types.Type{},
							Results:    []*types.Type{},
						},
					},
				},
			},
			expect: false,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Wrong signature (no result).
					"DeepCopy": {
						Name: types.Name{Package: "pkgname", Name: "func()"},
						Kind: types.Func,
						Signature: &types.Signature{
							Parameters: []*types.Type{},
							Results:    []*types.Type{},
						},
					},
				},
			},
			expect: false,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Wrong signature (wrong result).
					"DeepCopy": {
						Name: types.Name{Package: "pkgname", Name: "func() int"},
						Kind: types.Func,
						Signature: &types.Signature{
							Parameters: []*types.Type{},
							Results: []*types.Type{
								{
									Name: types.Name{Name: "int"},
									Kind: types.Builtin,
								},
							},
						},
					},
				},
			},
			expect: false,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Correct signature.
					"DeepCopy": {
						Name: types.Name{Package: "pkgname", Name: "func() pkgname.typename"},
						Kind: types.Func,
						Signature: &types.Signature{
							Parameters: []*types.Type{},
							Results: []*types.Type{
								{
									Name: types.Name{Package: "pkgname", Name: "typename"},
									Kind: types.Builtin,
								},
							},
						},
					},
				},
			},
			expect: true,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Wrong signature (has params).
					"DeepCopy": {
						Name: types.Name{Package: "pkgname", Name: "func(int) pkgname.typename"},
						Kind: types.Func,
						Signature: &types.Signature{
							Parameters: []*types.Type{
								{
									Name: types.Name{Name: "int"},
									Kind: types.Builtin,
								},
							},
							Results: []*types.Type{
								{
									Name: types.Name{Package: "pkgname", Name: "typename"},
									Kind: types.Builtin,
								},
							},
						},
					},
				},
			},
			expect: false,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Wrong signature (extra results).
					"DeepCopy": {
						Name: types.Name{Package: "pkgname", Name: "func() (pkgname.typename, int)"},
						Kind: types.Func,
						Signature: &types.Signature{
							Parameters: []*types.Type{},
							Results: []*types.Type{
								{
									Name: types.Name{Package: "pkgname", Name: "typename"},
									Kind: types.Builtin,
								},
								{
									Name: types.Name{Name: "int"},
									Kind: types.Builtin,
								},
							},
						},
					},
				},
			},
			expect: false,
		},
	}

	for i, tc := range testCases {
		r := hasDeepCopyMethod(&tc.typ)
		if r != tc.expect {
			t.Errorf("case[%d]: expected %t, got %t", i, tc.expect, r)
		}
	}
}

func Test_extractTagParams(t *testing.T) {
	testCases := []struct {
		comments []string
		expect   *tagValue
	}{
		{
			comments: []string{
				"Human comment",
			},
			expect: nil,
		},
		{
			comments: []string{
				"Human comment",
				"+k8s:deepcopy-gen",
			},
			expect: &tagValue{
				value:    "",
				register: false,
			},
		},
		{
			comments: []string{
				"Human comment",
				"+k8s:deepcopy-gen=package",
			},
			expect: &tagValue{
				value:    "package",
				register: false,
			},
		},
		{
			comments: []string{
				"Human comment",
				"+k8s:deepcopy-gen=package,register",
			},
			expect: &tagValue{
				value:    "package",
				register: true,
			},
		},
		{
			comments: []string{
				"Human comment",
				"+k8s:deepcopy-gen=package,register=true",
			},
			expect: &tagValue{
				value:    "package",
				register: true,
			},
		},
		{
			comments: []string{
				"Human comment",
				"+k8s:deepcopy-gen=package,register=false",
			},
			expect: &tagValue{
				value:    "package",
				register: false,
			},
		},
	}

	for i, tc := range testCases {
		r := extractTag(tc.comments)
		if r == nil && tc.expect != nil {
			t.Errorf("case[%d]: expected non-nil", i)
		}
		if r != nil && tc.expect == nil {
			t.Errorf("case[%d]: expected nil, got %v", i, *r)
		}
		if r != nil && *r != *tc.expect {
			t.Errorf("case[%d]: expected %v, got %v", i, *tc.expect, *r)
		}
	}
}

func Test_extractInterfacesTag(t *testing.T) {
	testCases := []struct {
		comments []string
		expect   []string
	}{
		{
			comments: []string{},
			expect: nil,
		},
		{
			comments: []string{
				"+k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/runtime.Object",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.Object",
			},
		},
		{
			comments: []string{
				"+k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/runtime.Object",
				"+k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/runtime.List",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.Object",
				"k8s.io/kubernetes/runtime.List",
			},
		},
		{
			comments: []string{
				"+k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/runtime.Object",
				"+k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/runtime.Object",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.Object",
				"k8s.io/kubernetes/runtime.Object",
			},
		},
	}

	for i, tc := range testCases {
		r := extractInterfacesTag(tc.comments)
		if r == nil && tc.expect != nil {
			t.Errorf("case[%d]: expected non-nil", i)
		}
		if r != nil && tc.expect == nil {
			t.Errorf("case[%d]: expected nil, got %v", i, r)
		}
		if r != nil && !reflect.DeepEqual(r, tc.expect) {
			t.Errorf("case[%d]: expected %v, got %v", i, tc.expect, r)
		}
	}
}