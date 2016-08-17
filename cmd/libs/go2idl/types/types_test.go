/*
Copyright 2015 The Kubernetes Authors.

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
	"testing"
)

func TestGetBuiltin(t *testing.T) {
	u := Universe{}
	if builtinPkg := u.Package(""); builtinPkg.Has("string") {
		t.Errorf("Expected builtin package to not have builtins until they're asked for explicitly. %#v", builtinPkg)
	}
	s := u.Type(Name{Package: "", Name: "string"})
	if s != String {
		t.Errorf("Expected canonical string type.")
	}
	if builtinPkg := u.Package(""); !builtinPkg.Has("string") {
		t.Errorf("Expected builtin package to exist and have builtins by default. %#v", builtinPkg)
	}
	if builtinPkg := u.Package(""); len(builtinPkg.Types) != 1 {
		t.Errorf("Expected builtin package to not have builtins until they're asked for explicitly. %#v", builtinPkg)
	}
}

func TestGetMarker(t *testing.T) {
	u := Universe{}
	n := Name{Package: "path/to/package", Name: "Foo"}
	f := u.Type(n)
	if f == nil || f.Name != n {
		t.Errorf("Expected marker type.")
	}
}

func Test_Type_IsPrimitive(t *testing.T) {
	testCases := []struct {
		typ    Type
		expect bool
	}{
		{
			typ: Type{
				Name: Name{Package: "pkgname", Name: "typename"},
				Kind: Builtin,
			},
			expect: true,
		},
		{
			typ: Type{
				Name: Name{Package: "pkgname", Name: "typename"},
				Kind: Alias,
				Underlying: &Type{
					Name: Name{Package: "pkgname", Name: "underlying"},
					Kind: Builtin,
				},
			},
			expect: true,
		},
		{
			typ: Type{
				Name: Name{Package: "pkgname", Name: "typename"},
				Kind: Pointer,
				Elem: &Type{
					Name: Name{Package: "pkgname", Name: "pointee"},
					Kind: Builtin,
				},
			},
			expect: false,
		},
		{
			typ: Type{
				Name: Name{Package: "pkgname", Name: "typename"},
				Kind: Struct,
			},
			expect: false,
		},
	}

	for i, tc := range testCases {
		r := tc.typ.IsPrimitive()
		if r != tc.expect {
			t.Errorf("case[%d]: expected %t, got %t", i, tc.expect, r)
		}
	}
}

func Test_Type_IsAssignable(t *testing.T) {
	testCases := []struct {
		typ    Type
		expect bool
	}{
		{
			typ: Type{
				Name: Name{Package: "pkgname", Name: "typename"},
				Kind: Builtin,
			},
			expect: true,
		},
		{
			typ: Type{
				Name: Name{Package: "pkgname", Name: "typename"},
				Kind: Alias,
				Underlying: &Type{
					Name: Name{Package: "pkgname", Name: "underlying"},
					Kind: Builtin,
				},
			},
			expect: true,
		},
		{
			typ: Type{
				Name: Name{Package: "pkgname", Name: "typename"},
				Kind: Pointer,
				Elem: &Type{
					Name: Name{Package: "pkgname", Name: "pointee"},
					Kind: Builtin,
				},
			},
			expect: false,
		},
		{
			typ: Type{
				Name: Name{Package: "pkgname", Name: "typename"},
				Kind: Struct, // Empty struct
			},
			expect: true,
		},
		{
			typ: Type{
				Name: Name{Package: "pkgname", Name: "typename"},
				Kind: Struct,
				Members: []Member{
					{
						Name: "m1",
						Type: &Type{
							Name: Name{Package: "pkgname", Name: "typename"},
							Kind: Builtin,
						},
					},
					{
						Name: "m2",
						Type: &Type{
							Name: Name{Package: "pkgname", Name: "typename"},
							Kind: Alias,
							Underlying: &Type{
								Name: Name{Package: "pkgname", Name: "underlying"},
								Kind: Builtin,
							},
						},
					},
					{
						Name: "m3",
						Type: &Type{
							Name: Name{Package: "pkgname", Name: "typename"},
							Kind: Struct, // Empty struct
						},
					},
				},
			},
			expect: true,
		},
		{
			typ: Type{
				Name: Name{Package: "pkgname", Name: "typename"},
				Kind: Struct,
				Members: []Member{
					{
						Name: "m1",
						Type: &Type{
							Name: Name{Package: "pkgname", Name: "typename"},
							Kind: Builtin,
						},
					},
					{
						Name: "m2",
						Type: &Type{
							Name: Name{Package: "pkgname", Name: "typename"},
							Kind: Pointer,
							Elem: &Type{
								Name: Name{Package: "pkgname", Name: "pointee"},
								Kind: Builtin,
							},
						},
					},
				},
			},
			expect: false,
		},
	}

	for i, tc := range testCases {
		r := tc.typ.IsAssignable()
		if r != tc.expect {
			t.Errorf("case[%d]: expected %t, got %t", i, tc.expect, r)
		}
	}
}
