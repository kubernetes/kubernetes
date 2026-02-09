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

package zerooroneof

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestZeroOrOneOfValidation(t *testing.T) {
	intPtr := func(i int) *int {
		return &i
	}

	tests := []struct {
		name string
		obj  interface{}
		errs field.ErrorList
	}{
		{
			name: "fails",
			obj: &ZeroOrOneOfStruct{
				Field1: intPtr(1),
				Field2: intPtr(2),
			},
			errs: field.ErrorList{
				field.Invalid(nil, map[string]interface{}{"field1": int(1), "field2": int(2)}, "must specify at most one of: `field1`, `field2`").MarkDeclarativeNative(),
			},
		},
		{
			name: "pass 1",
			obj: &ZeroOrOneOfStruct{
				Field1: intPtr(1),
			},
		},
		{
			name: "pass 2",
			obj: &ZeroOrOneOfStruct{
				Field2: intPtr(2),
			},
		},
		{
			name: "pass empty",
			obj:  &ZeroOrOneOfStruct{},
		},
		{
			name: "liststruct - fails",
			obj: &ZeroOrOneOfListStruct{
				Items: []UnionItem{
					{Type: "a", A: intPtr(1)},
					{Type: "b", B: intPtr(2)},
				},
			},
			errs: field.ErrorList{
				field.Invalid(field.NewPath("items"), []UnionItem{{Type: "a", A: intPtr(1)}, {Type: "b", B: intPtr(2)}}, "must specify at most one of: `items[{\"type\": \"a\"}]`, `items[{\"type\": \"b\"}]`").MarkDeclarativeNative(),
			},
		},
		{
			name: "liststruct - pass 1",
			obj: &ZeroOrOneOfListStruct{
				Items: []UnionItem{
					{Type: "a", A: intPtr(1)},
				},
			},
		},
		{
			name: "liststruct - pass 2",
			obj: &ZeroOrOneOfListStruct{
				Items: []UnionItem{
					{Type: "b", B: intPtr(2)},
				},
			},
		},
		{
			name: "liststruct - pass 3",
			obj: &ZeroOrOneOfListStruct{
				Items: []UnionItem{
					{Type: "c"},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			st := localSchemeBuilder.Test(t)
			st.Value(tc.obj).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDeclarativeNative(), tc.errs)
		})
	}
}

func TestFixtures(t *testing.T) {
	localSchemeBuilder.Test(t).ValidateFixtures()
}
