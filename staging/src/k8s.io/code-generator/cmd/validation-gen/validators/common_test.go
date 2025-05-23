/*
Copyright 2025 The Kubernetes Authors.

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
package validators

import (
	"testing"

	"k8s.io/gengo/v2/types"
)

// gengo has `PointerTo()` but not the rest, so keep this here for consistency.
func ptrTo(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "*" + t.Name.String(),
		},
		Kind: types.Pointer,
		Elem: t,
	}
}

func sliceOf(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "[]" + t.Name.String(),
		},
		Kind: types.Slice,
		Elem: t,
	}
}

func mapOf(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "map[string]" + t.Name.String(),
		},
		Kind: types.Map,
		Key:  types.String,
		Elem: t,
	}
}

func arrayOf(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "[2]" + t.Name.String(),
		},
		Kind: types.Array,
		Len:  2,
		Elem: t,
	}
}

func aliasOf(name string, t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "Alias_" + name,
		},
		Kind:       types.Alias,
		Underlying: t,
	}
}

func TestIsDirectComparable(t *testing.T) {
	cases := []struct {
		in     *types.Type
		expect bool
	}{
		{
			in:     types.String,
			expect: true,
		}, {
			in:     ptrTo(types.String),
			expect: false,
		}, {
			in:     sliceOf(types.String),
			expect: false,
		}, {
			in:     mapOf(types.String),
			expect: false,
		}, {
			in:     aliasOf("s", types.String),
			expect: true,
		}, {
			in: &types.Type{
				Name: types.Name{
					Package: "",
					Name:    "struct_comparable_member",
				},
				Kind: types.Struct,
				Members: []types.Member{
					{
						Name: "s",
						Type: types.String,
					},
				},
			},
			expect: true,
		}, {
			in: &types.Type{
				Name: types.Name{
					Package: "",
					Name:    "struct_uncomparable_member",
				},
				Kind: types.Struct,
				Members: []types.Member{
					{
						Name: "s",
						Type: ptrTo(types.String),
					},
				},
			},
			expect: false,
		}, {
			in:     arrayOf(types.String),
			expect: true,
		}, {
			in:     arrayOf(aliasOf("s", types.String)),
			expect: true,
		}, {
			in:     arrayOf(ptrTo(types.String)),
			expect: false,
		}, {
			in:     arrayOf(mapOf(types.String)),
			expect: false,
		},
	}

	for _, tc := range cases {
		if got, want := IsDirectComparable(tc.in), tc.expect; got != want {
			t.Errorf("%q: expected %v, got %v", tc.in, want, got)
		}
	}
}
