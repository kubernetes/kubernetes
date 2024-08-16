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

	"k8s.io/gengo/v2/types"
)

func Test_deepCopyMethod(t *testing.T) {
	testCases := []struct {
		typ    types.Type
		expect bool
		error  bool
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
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{},
							Results:    []*types.ParamResult{},
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
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{},
							Results:    []*types.ParamResult{},
						},
					},
				},
			},
			expect: false,
			error:  true,
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
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{},
							Results: []*types.ParamResult{
								{
									Type: &types.Type{Kind: types.Builtin, Name: types.Name{Name: "int"}},
								},
							},
						},
					},
				},
			},
			expect: false,
			error:  true,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Wrong signature with pointer receiver, but non-pointer result.
					"DeepCopy": {
						Name: types.Name{Package: "pkgname", Name: "func() pkgname.typename"},
						Kind: types.Func,
						Signature: &types.Signature{
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{},
							Results: []*types.ParamResult{
								{
									Type: &types.Type{Kind: types.Builtin, Name: types.Name{Package: "pkgname", Name: "typename"}},
								},
							},
						},
					},
				},
			},
			expect: false,
			error:  true,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Wrong signature with non-pointer receiver, but pointer result.
					"DeepCopy": {
						Name: types.Name{Package: "pkgname", Name: "func() pkgname.typename"},
						Kind: types.Func,
						Signature: &types.Signature{
							Receiver:   &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							Parameters: []*types.ParamResult{},
							Results: []*types.ParamResult{
								{
									Type: &types.Type{Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}}, Kind: types.Pointer},
								},
							},
						},
					},
				},
			},
			expect: false,
			error:  true,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Correct signature with non-pointer receiver.
					"DeepCopy": {
						Name: types.Name{Package: "pkgname", Name: "func() pkgname.typename"},
						Kind: types.Func,
						Signature: &types.Signature{
							Receiver:   &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							Parameters: []*types.ParamResult{},
							Results: []*types.ParamResult{
								{
									Type: &types.Type{Kind: types.Builtin, Name: types.Name{Package: "pkgname", Name: "typename"}},
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
					// Correct signature with pointer receiver.
					"DeepCopy": {
						Name: types.Name{Package: "pkgname", Name: "func() pkgname.typename"},
						Kind: types.Func,
						Signature: &types.Signature{
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{},
							Results: []*types.ParamResult{
								{
									Type: &types.Type{Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}}, Kind: types.Pointer},
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
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{
								{
									Type: &types.Type{Name: types.Name{Name: "int"}, Kind: types.Builtin},
								},
							},
							Results: []*types.ParamResult{
								{
									Type: &types.Type{Kind: types.Builtin, Name: types.Name{Package: "pkgname", Name: "typename"}},
								},
							},
						},
					},
				},
			},
			expect: false,
			error:  true,
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
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{},
							Results: []*types.ParamResult{
								{
									Type: &types.Type{Kind: types.Builtin, Name: types.Name{Package: "pkgname", Name: "typename"}},
								},
								{
									Type: &types.Type{Kind: types.Builtin, Name: types.Name{Name: "int"}},
								},
							},
						},
					},
				},
			},
			expect: false,
			error:  true,
		},
	}

	for i, tc := range testCases {
		r, err := deepCopyMethod(&tc.typ)
		if tc.error && err == nil {
			t.Errorf("case[%d]: expected an error, got none", i)
		} else if !tc.error && err != nil {
			t.Errorf("case[%d]: expected no error, got: %v", i, err)
		} else if !tc.error && (r != nil) != tc.expect {
			t.Errorf("case[%d]: expected result %v, got: %v", i, tc.expect, r)
		}
	}
}

func Test_deepCopyIntoMethod(t *testing.T) {
	testCases := []struct {
		typ    types.Type
		expect bool
		error  bool
	}{
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				// No DeepCopyInto method.
				Methods: map[string]*types.Type{},
			},
			expect: false,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// No DeepCopyInto method.
					"method": {
						Name: types.Name{Package: "pkgname", Name: "func()"},
						Kind: types.Func,
						Signature: &types.Signature{
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{},
							Results:    []*types.ParamResult{},
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
					// Wrong signature (no parameter).
					"DeepCopyInto": {
						Name: types.Name{Package: "pkgname", Name: "func()"},
						Kind: types.Func,
						Signature: &types.Signature{
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{},
							Results:    []*types.ParamResult{},
						},
					},
				},
			},
			expect: false,
			error:  true,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Wrong signature (unexpected result).
					"DeepCopyInto": {
						Name: types.Name{Package: "pkgname", Name: "func(*pkgname.typename) int"},
						Kind: types.Func,
						Signature: &types.Signature{
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{
								{
									Type: &types.Type{Kind: types.Pointer, Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}}},
								},
							},
							Results: []*types.ParamResult{
								{
									Type: &types.Type{Kind: types.Builtin, Name: types.Name{Name: "int"}},
								},
							},
						},
					},
				},
			},
			expect: false,
			error:  true,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Wrong signature (non-pointer parameter, pointer receiver).
					"DeepCopyInto": {
						Name: types.Name{Package: "pkgname", Name: "func(pkgname.typename)"},
						Kind: types.Func,
						Signature: &types.Signature{
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{
								{
									Type: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
								},
							},
							Results: []*types.ParamResult{},
						},
					},
				},
			},
			expect: false,
			error:  true,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Wrong signature (non-pointer parameter, non-pointer receiver).
					"DeepCopyInto": {
						Name: types.Name{Package: "pkgname", Name: "func(pkgname.typename)"},
						Kind: types.Func,
						Signature: &types.Signature{
							Receiver: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							Parameters: []*types.ParamResult{
								{
									Type: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
								},
							},
							Results: []*types.ParamResult{},
						},
					},
				},
			},
			expect: false,
			error:  true,
		},
		{
			typ: types.Type{
				Name: types.Name{Package: "pkgname", Name: "typename"},
				Kind: types.Builtin,
				Methods: map[string]*types.Type{
					// Correct signature with non-pointer receiver.
					"DeepCopyInto": {
						Name: types.Name{Package: "pkgname", Name: "func(*pkgname.typename)"},
						Kind: types.Func,
						Signature: &types.Signature{
							Receiver: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							Parameters: []*types.ParamResult{
								{
									Type: &types.Type{Kind: types.Pointer, Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}}},
								},
							},
							Results: []*types.ParamResult{},
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
					// Correct signature with pointer receiver.
					"DeepCopyInto": {
						Name: types.Name{Package: "pkgname", Name: "func(*pkgname.typename)"},
						Kind: types.Func,
						Signature: &types.Signature{
							Receiver: &types.Type{
								Kind: types.Pointer,
								Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}},
							},
							Parameters: []*types.ParamResult{
								{
									Type: &types.Type{Kind: types.Pointer, Elem: &types.Type{Kind: types.Struct, Name: types.Name{Package: "pkgname", Name: "typename"}}},
								},
							},
							Results: []*types.ParamResult{},
						},
					},
				},
			},
			expect: true,
		},
	}

	for i, tc := range testCases {
		r, err := deepCopyIntoMethod(&tc.typ)
		if tc.error && err == nil {
			t.Errorf("case[%d]: expected an error, got none", i)
		} else if !tc.error && err != nil {
			t.Errorf("case[%d]: expected no error, got: %v", i, err)
		} else if !tc.error && (r != nil) != tc.expect {
			t.Errorf("case[%d]: expected result %v, got: %v", i, tc.expect, r)
		}
	}
}

func Test_extractTagParams(t *testing.T) {
	testCases := []struct {
		comments []string
		expect   *enabledTagValue
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
			expect: &enabledTagValue{
				value:    "",
				register: false,
			},
		},
		{
			comments: []string{
				"Human comment",
				"+k8s:deepcopy-gen=package",
			},
			expect: &enabledTagValue{
				value:    "package",
				register: false,
			},
		},
		{
			comments: []string{
				"Human comment",
				"+k8s:deepcopy-gen=package,register",
			},
			expect: &enabledTagValue{
				value:    "package",
				register: true,
			},
		},
		{
			comments: []string{
				"Human comment",
				"+k8s:deepcopy-gen=package,register=true",
			},
			expect: &enabledTagValue{
				value:    "package",
				register: true,
			},
		},
		{
			comments: []string{
				"Human comment",
				"+k8s:deepcopy-gen=package,register=false",
			},
			expect: &enabledTagValue{
				value:    "package",
				register: false,
			},
		},
	}

	for i, tc := range testCases {
		r := extractEnabledTag(tc.comments)
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
		comments, secondComments []string
		expect                   []string
	}{
		{
			comments: []string{},
			expect:   nil,
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
		{
			secondComments: []string{
				"+k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/runtime.Object",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.Object",
			},
		},
		{
			comments: []string{
				"+k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/runtime.Object",
			},
			secondComments: []string{
				"+k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/runtime.List",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.List",
				"k8s.io/kubernetes/runtime.Object",
			},
		},
		{
			comments: []string{
				"+k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/runtime.Object",
			},
			secondComments: []string{
				"+k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/runtime.Object",
			},
			expect: []string{
				"k8s.io/kubernetes/runtime.Object",
				"k8s.io/kubernetes/runtime.Object",
			},
		},
	}

	for i, tc := range testCases {
		typ := &types.Type{
			CommentLines:              tc.comments,
			SecondClosestCommentLines: tc.secondComments,
		}
		r := extractInterfacesTag(typ)
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
