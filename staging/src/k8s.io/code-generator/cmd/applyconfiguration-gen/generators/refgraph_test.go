/*
Copyright The Kubernetes Authors.

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

func TestRequiresApplyConfiguration(t *testing.T) {
	testCases := []struct {
		name     string
		typ      *types.Type
		expected bool
	}{
		{
			name: "struct with json tagged fields",
			typ: &types.Type{
				Name: types.Name{Package: "example.com/api/v1", Name: "StandardStruct"},
				Kind: types.Struct,
				Members: []types.Member{
					{Name: "FieldA", Tags: `json:"fieldA,omitempty"`},
				},
			},
			expected: true,
		},
		{
			name: "struct with json tagged fields and custom MarshalJSON method",
			typ: &types.Type{
				Name: types.Name{Package: "example.com/api/v1", Name: "CustomMarshalStruct"},
				Kind: types.Struct,
				Members: []types.Member{
					{Name: "Host", Tags: `json:"host,omitempty"`},
					{Name: "Port", Tags: `json:"port,omitempty"`},
				},
				Methods: map[string]*types.Type{
					"MarshalJSON": {Kind: types.Func},
				},
			},
			expected: false,
		},
		{
			name: "struct with json tagged fields and custom UnmarshalJSON method",
			typ: &types.Type{
				Name: types.Name{Package: "example.com/api/v1", Name: "CustomUnmarshalStruct"},
				Kind: types.Struct,
				Members: []types.Member{
					{Name: "Host", Tags: `json:"host,omitempty"`},
					{Name: "Port", Tags: `json:"port,omitempty"`},
				},
				Methods: map[string]*types.Type{
					"UnmarshalJSON": {Kind: types.Func},
				},
			},
			expected: false,
		},
		{
			name: "struct with json tagged fields and both MarshalJSON and UnmarshalJSON methods",
			typ: &types.Type{
				Name: types.Name{Package: "example.com/api/v1", Name: "CustomBothJSONStruct"},
				Kind: types.Struct,
				Members: []types.Member{
					{Name: "Host", Tags: `json:"host,omitempty"`},
					{Name: "Port", Tags: `json:"port,omitempty"`},
				},
				Methods: map[string]*types.Type{
					"MarshalJSON":   {Kind: types.Func},
					"UnmarshalJSON": {Kind: types.Func},
				},
			},
			expected: false,
		},
		{
			name: "struct without json tagged fields",
			typ: &types.Type{
				Name: types.Name{Package: "example.com/api/v1", Name: "NoJSONTags"},
				Kind: types.Struct,
				Members: []types.Member{
					{Name: "FieldA", Tags: `protobuf:"bytes,1,opt,name=fieldA"`},
				},
			},
			expected: false,
		},
		{
			name: "type in excludeTypes",
			typ: &types.Type{
				Name: rawExtension.Name,
				Kind: types.Struct,
				Members: []types.Member{
					{Name: "Raw", Tags: `json:"raw,omitempty"`},
				},
			},
			expected: false,
		},
		{
			name: "non-struct type",
			typ: &types.Type{
				Name: types.Name{Package: "example.com/api/v1", Name: "SomeString"},
				Kind: types.Builtin,
			},
			expected: false,
		},
		{
			name: "alias to struct with json tags",
			typ: &types.Type{
				Name: types.Name{Package: "example.com/api/v1", Name: "AliasToStandard"},
				Kind: types.Alias,
				Underlying: &types.Type{
					Name: types.Name{Package: "example.com/api/v1", Name: "StandardStruct"},
					Kind: types.Struct,
					Members: []types.Member{
						{Name: "FieldA", Tags: `json:"fieldA,omitempty"`},
					},
				},
			},
			expected: true,
		},
		{
			name: "alias with custom MarshalJSON to struct with json tags",
			typ: &types.Type{
				Name: types.Name{Package: "example.com/api/v1", Name: "AliasWithMarshal"},
				Kind: types.Alias,
				Methods: map[string]*types.Type{
					"MarshalJSON": {Kind: types.Func},
				},
				Underlying: &types.Type{
					Name: types.Name{Package: "example.com/api/v1", Name: "StandardStruct"},
					Kind: types.Struct,
					Members: []types.Member{
						{Name: "FieldA", Tags: `json:"fieldA,omitempty"`},
					},
				},
			},
			expected: false,
		},
		{
			name: "alias to struct with custom MarshalJSON",
			typ: &types.Type{
				Name: types.Name{Package: "example.com/api/v1", Name: "AliasToCustomStruct"},
				Kind: types.Alias,
				Underlying: &types.Type{
					Name: types.Name{Package: "example.com/api/v1", Name: "CustomMarshalStruct"},
					Kind: types.Struct,
					Members: []types.Member{
						{Name: "Host", Tags: `json:"host,omitempty"`},
						{Name: "Port", Tags: `json:"port,omitempty"`},
					},
					Methods: map[string]*types.Type{
						"MarshalJSON": {Kind: types.Func},
					},
				},
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := requiresApplyConfiguration(tc.typ)
			if got != tc.expected {
				t.Errorf("requiresApplyConfiguration(%s) = %v, want %v", tc.name, got, tc.expected)
			}
		})
	}
}

func TestApplyConfigForTypeWithCustomJSONSerialization(t *testing.T) {
	customJSONStruct := &types.Type{
		Name: types.Name{Package: "example.com/api/v1", Name: "Address"},
		Kind: types.Struct,
		Members: []types.Member{
			{Name: "Host", Tags: `json:"host,omitempty"`},
			{Name: "Port", Tags: `json:"port,omitempty"`},
		},
		Methods: map[string]*types.Type{
			"MarshalJSON":   {Kind: types.Func},
			"UnmarshalJSON": {Kind: types.Func},
		},
	}

	standardStruct := &types.Type{
		Name: types.Name{Package: "example.com/api/v1", Name: "Standard"},
		Kind: types.Struct,
		Members: []types.Member{
			{Name: "Field", Tags: `json:"field,omitempty"`},
		},
	}

	// refGraph only registers types that requiresApplyConfiguration() is true for.
	refs := refGraph{}
	if requiresApplyConfiguration(standardStruct) {
		refs[standardStruct.Name] = "example.com/applyconfiguration/api/v1"
	}
	if requiresApplyConfiguration(customJSONStruct) {
		refs[customJSONStruct.Name] = "example.com/applyconfiguration/api/v1"
	}

	// standardStruct should resolve to an ApplyConfiguration type
	if !refs.isApplyConfig(standardStruct) {
		t.Errorf("expected standardStruct to be recognized as apply configuration")
	}
	gotStandard := refs.applyConfigForType(standardStruct)
	wantStandard := types.Name{
		Package: "example.com/applyconfiguration/api/v1",
		Name:    "StandardApplyConfiguration",
	}
	if gotStandard.Name != wantStandard {
		t.Errorf("got %v, want %v", gotStandard.Name, wantStandard)
	}

	// customJSONStruct should resolve to the original Go type as a whole value
	if refs.isApplyConfig(customJSONStruct) {
		t.Errorf("expected customJSONStruct to NOT be recognized as apply configuration")
	}
	gotCustom := refs.applyConfigForType(customJSONStruct)
	if !reflect.DeepEqual(gotCustom, customJSONStruct) {
		t.Errorf("got %v, want original type %v", gotCustom, customJSONStruct)
	}

	// Pointer to customJSONStruct should resolve to unwrapped original Go type (by design, applyConfigForType unwraps pointers)
	ptrCustom := &types.Type{
		Kind: types.Pointer,
		Elem: customJSONStruct,
	}
	gotPtr := refs.applyConfigForType(ptrCustom)
	if !reflect.DeepEqual(gotPtr, customJSONStruct) {
		t.Errorf("got %v, want %v", gotPtr, customJSONStruct)
	}

	// Slice of customJSONStruct should resolve to slice of original Go type
	sliceCustom := &types.Type{
		Kind: types.Slice,
		Elem: customJSONStruct,
	}
	gotSlice := refs.applyConfigForType(sliceCustom)
	if gotSlice.Kind != types.Slice || !reflect.DeepEqual(gotSlice.Elem, customJSONStruct) {
		t.Errorf("got %v, want slice of %v", gotSlice, customJSONStruct)
	}

	// Map of customJSONStruct should resolve to map of original Go type
	mapCustom := &types.Type{
		Kind: types.Map,
		Key:  &types.Type{Kind: types.Builtin, Name: types.Name{Name: "string"}},
		Elem: customJSONStruct,
	}
	gotMap := refs.applyConfigForType(mapCustom)
	if gotMap.Kind != types.Map || !reflect.DeepEqual(gotMap.Elem, customJSONStruct) {
		t.Errorf("got %v, want map of %v", gotMap, customJSONStruct)
	}
}
