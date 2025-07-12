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

package typedef

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		TypedefItems: ItemList{
			{Key: "a", Data: "d1"},
			{Key: "b", Data: "d2"},
		},
	}).ExpectValid()

	st.Value(&Struct{
		TypedefItems: ItemList{
			{Key: "a", Data: "d1"},
			{Key: "validated", Data: "d2"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		`typedefItems[1]`: {"item ItemList[key=validated]"},
	})

	// Test immutability on typedef.
	oldStruct := &Struct{
		TypedefItems: ItemList{
			{Key: "immutable", Data: "original"},
		},
	}
	newStruct := &Struct{
		TypedefItems: ItemList{
			{Key: "immutable", Data: "changed"},
		},
	}
	st.Value(newStruct).OldValue(oldStruct).ExpectInvalid(
		field.Forbidden(field.NewPath("typedefItems").Index(0), "field is immutable"),
	)

	// Test nested typedef (typedef of typedef).
	st.Value(&Struct{
		NestedTypedefItems: ItemListAlias{
			{Key: "normal", Data: "d1"},
			{Key: "aliased", Data: "d2"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		`nestedTypedefItems[1]`: {"item ItemListAlias[key=aliased]"},
	})

	// Test tag on field and typedef.
	st.Value(&Struct{
		DualItems: DualItemList{
			{ID: "a", Name: "n1"},
			{ID: "typedef-target", Name: "n2"},
			{ID: "field-target", Name: "n3"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		`dualItems[1]`: {"item DualItems[id=typedef-target] from typedef"},
		`dualItems[2]`: {"item DualItems[id=field-target] from field"},
	})
}
