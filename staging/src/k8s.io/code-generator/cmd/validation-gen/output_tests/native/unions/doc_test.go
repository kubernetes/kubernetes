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

package unions

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestUnionValidation(t *testing.T) {
	intPtr := func(i int) *int {
		return &i
	}

	tests := []struct {
		name string
		obj  interface{}
		errs field.ErrorList
	}{
		{
			name: "struct - union fails",
			obj:  &Struct{},
			errs: field.ErrorList{
				field.Invalid(nil, "", "must specify one of: `unionField1`, `unionField2`").MarkDeclarativeNative(),
			},
		},
		{
			name: "struct - union pass 1",
			obj: &Struct{
				UnionField1: intPtr(1),
			},
		},
		{
			name: "struct - union pass 2",
			obj: &Struct{
				UnionField2: intPtr(2),
			},
		},
		{
			name: "liststruct - union fails",
			obj: &ListStruct{
				Items: []UnionItem{
					{Type: "a", A: intPtr(1)},
					{Type: "b", B: intPtr(2)},
				},
			},
			errs: field.ErrorList{
				field.Invalid(field.NewPath("items"), []UnionItem{{Type: "a", A: intPtr(1)}, {Type: "b", B: intPtr(2)}}, "must specify one of: `items[{\"type\": \"a\"}]`, `items[{\"type\": \"b\"}]`").MarkDeclarativeNative(),
			},
		},
		{
			name: "liststruct - union pass 1",
			obj: &ListStruct{
				Items: []UnionItem{
					{Type: "a", A: intPtr(1)},
				},
			},
		},
		{
			name: "liststruct - union pass 2",
			obj: &ListStruct{
				Items: []UnionItem{
					{Type: "b", B: intPtr(2)},
				},
			},
		},
		{
			name: "liststruct - union fails (none set)",
			obj: &ListStruct{
				Items: []UnionItem{
					{Type: "c"},
				},
			},
			errs: field.ErrorList{
				field.Invalid(field.NewPath("items"), []UnionItem{{Type: "c"}}, "must specify one of: `items[{\"type\": \"a\"}]`, `items[{\"type\": \"b\"}]`").MarkDeclarativeNative(),
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
