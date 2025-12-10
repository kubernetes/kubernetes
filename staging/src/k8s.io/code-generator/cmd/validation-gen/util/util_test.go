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

package util

import (
	"reflect"
	"testing"

	"k8s.io/gengo/v2/types"
)

func TestGetMemberByJSON(t *testing.T) {
	tests := []struct {
		name     string
		t        *types.Type
		jsonTag  string
		want     *types.Member
		wantBool bool
	}{{
		name: "exact match",
		t: &types.Type{
			Members: []types.Member{
				{Name: "Field0", Tags: `json:"field0"`},
				{Name: "Field1", Tags: `json:"field1"`},
				{Name: "Field2", Tags: `json:"field2"`},
			},
		},
		jsonTag:  "field1",
		want:     &types.Member{Name: "Field1", Tags: `json:"field1"`},
		wantBool: true,
	}, {
		name: "no match",
		t: &types.Type{
			Members: []types.Member{
				{Name: "Field0", Tags: `json:"field0"`},
				{Name: "Field1", Tags: `json:"field1"`},
			},
		},
		jsonTag:  "field2",
		want:     nil,
		wantBool: false,
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetMemberByJSON(tt.t, tt.jsonTag)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetMemberByJSON() got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsNilableType(t *testing.T) {
	tStruct := &types.Type{
		Name: types.Name{Name: "MyStruct"},
		Kind: types.Struct,
	}

	tests := []struct {
		name string
		t    *types.Type
		want bool
	}{{
		name: "pointer",
		t: &types.Type{
			Kind: types.Pointer,
			Elem: tStruct,
		},
		want: true,
	}, {
		name: "alias to pointer",
		t: &types.Type{
			Kind: types.Alias,
			Underlying: &types.Type{
				Kind: types.Pointer,
				Elem: tStruct,
			},
		},
		want: true,
	}, {
		name: "map",
		t: &types.Type{
			Kind: types.Map,
		},
		want: true,
	}, {
		name: "alias to map",
		t: &types.Type{
			Kind: types.Alias,
			Underlying: &types.Type{
				Kind: types.Map,
			},
		},
		want: true,
	}, {
		name: "slice",
		t: &types.Type{
			Kind: types.Slice,
		},
		want: true,
	}, {
		name: "alias to slice",
		t: &types.Type{
			Kind: types.Alias,
			Underlying: &types.Type{
				Kind: types.Slice,
			},
		},
		want: true,
	}, {
		name: "interface",
		t: &types.Type{
			Kind: types.Interface,
		},
		want: true,
	}, {
		name: "alias to interface",
		t: &types.Type{
			Kind: types.Alias,
			Underlying: &types.Type{
				Kind: types.Interface,
			},
		},
		want: true,
	}, {
		name: "struct",
		t: &types.Type{
			Kind: types.Struct,
		},
		want: false,
	}, {
		name: "alias to struct",
		t: &types.Type{
			Kind: types.Alias,
			Underlying: &types.Type{
				Kind: types.Struct,
			},
		},
		want: false,
	}, {
		name: "builtin",
		t: &types.Type{
			Kind: types.Builtin,
		},
		want: false,
	}, {
		name: "alias to builtin",
		t: &types.Type{
			Kind: types.Alias,
			Underlying: &types.Type{
				Kind: types.Builtin,
			},
		},
		want: false,
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsNilableType(tt.t); got != tt.want {
				t.Errorf("IsNilableType() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNativeType(t *testing.T) {
	tStruct := &types.Type{
		Name: types.Name{Name: "MyStruct"},
		Kind: types.Struct,
	}
	pStruct := &types.Type{
		Name: types.Name{Name: "*MyStruct"},
		Kind: types.Pointer,
		Elem: tStruct,
	}

	tests := []struct {
		name string
		t    *types.Type
		want *types.Type
	}{{
		name: "struct",
		t:    tStruct,
		want: tStruct,
	}, {
		name: "pointer to struct",
		t:    pStruct,
		want: pStruct,
	}, {
		name: "alias to struct",
		t: &types.Type{
			Name:       types.Name{Name: "Alias"},
			Kind:       types.Alias,
			Underlying: tStruct,
		},
		want: tStruct,
	}, {
		name: "pointer to alias to struct",
		t: &types.Type{
			Name: types.Name{Name: "*Alias"},
			Kind: types.Pointer,
			Elem: &types.Type{
				Kind:       types.Alias,
				Underlying: tStruct,
			},
		},
		want: pStruct,
	}, {
		name: "alias of pointer to struct",
		t: &types.Type{
			Name:       types.Name{Name: "AliasP"},
			Kind:       types.Alias,
			Underlying: pStruct,
		},
		want: pStruct,
	}, {
		name: "pointer to alias of pointer to struct",
		t: &types.Type{
			Name: types.Name{Name: "*AliasP"},
			Kind: types.Pointer,
			Elem: &types.Type{
				Name:       types.Name{Name: "AliasP"},
				Kind:       types.Alias,
				Underlying: pStruct,
			},
		},
		want: &types.Type{
			Name: types.Name{Name: "**MyStruct"},
			Kind: types.Pointer,
			Elem: pStruct,
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if want, got := tt.want.String(), NativeType(tt.t).String(); want != got {
				t.Errorf("NativeType() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNonPointer(t *testing.T) {
	tStruct := &types.Type{
		Name: types.Name{Name: "MyStruct"},
		Kind: types.Struct,
	}

	tests := []struct {
		name string
		t    *types.Type
		want *types.Type
	}{{
		name: "value",
		t:    tStruct,
		want: tStruct,
	}, {
		name: "pointer",
		t: &types.Type{
			Name: types.Name{Name: "*MyStruct"},
			Kind: types.Pointer,
			Elem: tStruct,
		},
		want: tStruct,
	}, {
		name: "pointer pointer",
		t: &types.Type{
			Name: types.Name{Name: "**MyStruct"},
			Kind: types.Pointer,
			Elem: &types.Type{
				Kind: types.Pointer,
				Elem: tStruct,
			},
		},
		want: tStruct,
	}, {
		name: "pointer alias pointer",
		t: &types.Type{
			Name: types.Name{Name: "*AliasP"},
			Kind: types.Pointer,
			Elem: &types.Type{
				Name: types.Name{Name: "AliasP"},
				Kind: types.Alias,
				Underlying: &types.Type{
					Name: types.Name{Name: "*MyStruct"},
					Kind: types.Pointer,
					Elem: tStruct,
				},
			},
		},
		want: &types.Type{
			Name: types.Name{Name: "AliasP"},
			Kind: types.Alias,
			Underlying: &types.Type{
				Name: types.Name{Name: "*MyStruct"},
				Kind: types.Pointer,
				Elem: tStruct,
			},
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NonPointer(tt.t); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NonPointer() = %v, want %v", got, tt.want)
			}
		})
	}
}

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
