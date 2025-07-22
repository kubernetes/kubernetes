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

package singlekey

import (
	"testing"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		Items: []Item{
			{Key: "a", Data: "d1"},
			{Key: "target", Data: "d2"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		`items[1]`: {"item Items[key=target]"},
	})

	st.Value(&Struct{
		Items: []Item{
			{Key: "a", Data: "d1"},
			{Key: "b", Data: "d2"},
		},
	}).ExpectValid()

	st.Value(&Struct{
		Items: []Item{},
	}).ExpectValid()

	st.Value(&Struct{
		Items: nil,
	}).ExpectValid()

	oldStruct := &Struct{Items: nil}
	newStruct := &Struct{Items: []Item{}}
	st.Value(newStruct).OldValue(oldStruct).ExpectValid()
	st.Value(oldStruct).OldValue(newStruct).ExpectValid()

	st.Value(&Struct{
		IntKeyItems: []IntKeyItem{
			{IntField: 10, Data: "d1"},
			{IntField: 42, Data: "d2"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		`intKeyItems[1]`: {"item IntKeyItems[intField=42]"},
	})

	st.Value(&Struct{
		BoolKeyItems: []BoolKeyItem{
			{BoolField: false, Data: "d1"},
			{BoolField: true, Data: "d2"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		`boolKeyItems[1]`: {"item BoolKeyItems[boolField=true]"},
	})

	// Test typedef slice.
	st.Value(&Struct{
		TypedefItems: TypedefItemList{
			{ID: "a", Description: "d1"},
			{ID: "typedef-target", Description: "d2"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		`typedefItems[1]`: {"item TypedefItems[id=typedef-target]"},
	})

	st.Value(&Struct{
		TypedefItems: TypedefItemList{
			{ID: "a", Description: "d1"},
			{ID: "b", Description: "d2"},
		},
	}).ExpectValid()

	// Test nested typedef.
	st.Value(&StructWithNestedTypedef{
		NestedItems: []NestedTypedefItem{
			{Key: "a", Name: "n1"},
			{Key: "nested-target", Name: "n2"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		`nestedItems[1]`: {"item NestedItems[key=nested-target]"},
	})

	st.Value(&StructWithNestedTypedef{
		NestedItems: []NestedTypedefItem{
			{Key: "a", Name: "n1"},
			{Key: "b", Name: "n2"},
		},
	}).ExpectValid()
}
