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

package multiplekeys

import (
	"testing"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

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
		Items: []Item{
			{StringKey: "target", IntKey: 42, BoolKey: true, Data: "match"},
			{StringKey: "target", IntKey: 42, BoolKey: false, Data: "no match, bool differs"},
			{StringKey: "target", IntKey: 99, BoolKey: true, Data: "no match, int differs"},
			{StringKey: "other", IntKey: 42, BoolKey: true, Data: "no match, string differs"},
			{StringKey: "other", IntKey: 99, BoolKey: false, Data: "no match, all different"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		`items[0]`: {"item Items[stringKey=target,intKey=42,boolKey=true]"},
	})

	st.Value(&Struct{
		Items: []Item{
			{StringKey: "a", IntKey: 1, BoolKey: false, Data: "d1"},
			{StringKey: "b", IntKey: 2, BoolKey: true, Data: "d2"},
			{StringKey: "c", IntKey: 3, BoolKey: false, Data: "d3"},
		},
	}).ExpectValid()

	// Test ratcheting.
	st.Value(&Struct{
		Items: []Item{
			{StringKey: "target", IntKey: 42, BoolKey: true},
			{StringKey: "changed", IntKey: 2, BoolKey: false},
		},
	}).OldValue(&Struct{
		Items: []Item{
			{StringKey: "target", IntKey: 42, BoolKey: true},
			{StringKey: "other", IntKey: 1, BoolKey: false},
		},
	}).ExpectValid()
}
