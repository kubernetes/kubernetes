/*
Copyright 2024 The Kubernetes Authors.

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

package sliceofstruct

import (
	"testing"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values.
	}).ExpectValid()

	st.Value(&Struct{
		ListField:        []OtherStruct{{}, {}},
		ListTypedefField: []OtherTypedefStruct{{}, {}},
	}).ExpectValidateFalseByPath(map[string][]string{
		"listField[0]":        {"field Struct.ListField[*]"},
		"listField[1]":        {"field Struct.ListField[*]"},
		"listTypedefField[0]": {"field Struct.ListTypedefField[*]"},
		"listTypedefField[1]": {"field Struct.ListTypedefField[*]"},
	})
	st.Value(&Struct{
		ListNonComparableField: []NonComparableStruct{{SliceField: []string{"zero", "one"}}, {SliceField: []string{"three", "four"}}},
	}).ExpectValidateFalseByPath(map[string][]string{
		"listNonComparableField[0]": {"field Struct.ListNonComparableField[*]"},
		"listNonComparableField[1]": {"field Struct.ListNonComparableField[*]"},
	})

	// Test validation ratcheting.
	st.Value(&Struct{
		ListField:        []OtherStruct{{}, {}},
		ListTypedefField: []OtherTypedefStruct{{}, {}},
	}).OldValue(&Struct{
		ListField:        []OtherStruct{{}, {}},
		ListTypedefField: []OtherTypedefStruct{{}, {}},
	}).ExpectValid()

	st.Value(&Struct{
		// New element exists in old value, but this is not a set.
		ListNonComparableField: []NonComparableStruct{{SliceField: []string{"three", "four"}}},
	}).OldValue(&Struct{
		ListNonComparableField: []NonComparableStruct{{SliceField: []string{"zero", "one"}}, {SliceField: []string{"three", "four"}}},
	}).ExpectValidateFalseByPath(map[string][]string{
		"listNonComparableField[0]": {"field Struct.ListNonComparableField[*]"},
	})
}
