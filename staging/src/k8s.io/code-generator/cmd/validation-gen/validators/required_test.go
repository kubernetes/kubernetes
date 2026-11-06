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

package validators

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

// A field context is documented to carry the struct member it describes.
// A caller which omits it must get an error, not a nil dereference.
func TestRequirednessWithoutMember(t *testing.T) {
	fieldType := &types.Type{Name: types.Name{Name: "int"}, Kind: types.Builtin}
	tv := requirednessTagValidator{mode: requirednessOptional}
	context := Context{Scope: ScopeField, Type: fieldType}

	_, err := tv.GetValidations(context, codetags.Tag{Name: string(requirednessOptional)})
	if err == nil {
		t.Fatal("GetValidations() succeeded, want an error")
	}
	if !strings.Contains(err.Error(), "no member") {
		t.Errorf("GetValidations() error = %v, want it to mention a missing member", err)
	}
}

func TestRequirednessPayloads(t *testing.T) {
	intType := &types.Type{Name: types.Name{Name: "int"}, Kind: types.Builtin}
	ptrType := &types.Type{Kind: types.Pointer, Elem: intType}
	sliceType := &types.Type{Kind: types.Slice, Elem: intType}
	mapType := &types.Type{Kind: types.Map, Key: intType, Elem: intType}
	structType := &types.Type{Name: types.Name{Name: "S"}, Kind: types.Struct}

	testCases := []struct {
		name         string
		mode         requirednessMode
		fieldType    *types.Type
		payload      string
		expectedArgs [][]any
	}{{
		name:         "required value without payload",
		mode:         requirednessRequired,
		fieldType:    intType,
		payload:      "",
		expectedArgs: [][]any{nil},
	}, {
		name:         "required value with payload",
		mode:         requirednessRequired,
		fieldType:    intType,
		payload:      "must be set",
		expectedArgs: [][]any{{"must be set"}},
	}, {
		name:         "required pointer with payload",
		mode:         requirednessRequired,
		fieldType:    ptrType,
		payload:      "must be set",
		expectedArgs: [][]any{{"must be set"}},
	}, {
		name:         "required slice with payload",
		mode:         requirednessRequired,
		fieldType:    sliceType,
		payload:      "must be set",
		expectedArgs: [][]any{{"must be set"}},
	}, {
		name:         "required map with payload",
		mode:         requirednessRequired,
		fieldType:    mapType,
		payload:      "must be set",
		expectedArgs: [][]any{{"must be set"}},
	}, {
		name:         "required non-pointer struct with payload is doc-only",
		mode:         requirednessRequired,
		fieldType:    structType,
		payload:      "must be set",
		expectedArgs: nil,
	}, {
		name:         "forbidden value without payload",
		mode:         requirednessForbidden,
		fieldType:    intType,
		payload:      "",
		expectedArgs: [][]any{nil, nil},
	}, {
		name:         "forbidden value with payload",
		mode:         requirednessForbidden,
		fieldType:    intType,
		payload:      "may not be set",
		expectedArgs: [][]any{{"may not be set"}, nil},
	}, {
		name:         "forbidden pointer with payload",
		mode:         requirednessForbidden,
		fieldType:    ptrType,
		payload:      "may not be set",
		expectedArgs: [][]any{{"may not be set"}, nil},
	}, {
		name:         "forbidden slice with payload",
		mode:         requirednessForbidden,
		fieldType:    sliceType,
		payload:      "may not be set",
		expectedArgs: [][]any{{"may not be set"}, nil},
	}, {
		name:         "forbidden map with payload",
		mode:         requirednessForbidden,
		fieldType:    mapType,
		payload:      "may not be set",
		expectedArgs: [][]any{{"may not be set"}, nil},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tv := requirednessTagValidator{mode: tc.mode, prefix: "k8s:"}
			tag := codetags.Tag{Name: "k8s:" + string(tc.mode)}
			if tc.payload != "" {
				tag.Value = tc.payload
				tag.ValueType = codetags.ValueTypeString
			}
			if err := typeCheck(tag, tv.Docs()); err != nil {
				t.Fatalf("typeCheck() failed: %v", err)
			}
			validations, err := tv.GetValidations(Context{Scope: ScopeField, Type: tc.fieldType}, tag)
			if err != nil {
				t.Fatalf("GetValidations() failed: %v", err)
			}
			var gotArgs [][]any
			for _, fn := range validations.Functions {
				gotArgs = append(gotArgs, fn.Args)
			}
			if want, got := tc.expectedArgs, gotArgs; !reflect.DeepEqual(got, want) {
				t.Errorf("expected %v, got %v", want, got)
			}
		})
	}
}
