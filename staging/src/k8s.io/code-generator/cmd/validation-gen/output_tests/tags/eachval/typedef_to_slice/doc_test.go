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

package typedeftoslice

import (
	"testing"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values.
	}).ExpectValid()

	st.Value(&Struct{
		ListField:        ListType{"zero", "one"},
		ListTypedefField: ListTypedefType{StringType("zero"), StringType("one")},
	}).ExpectValidateFalseByPath(map[string][]string{
		"listField[0]":        {"type ListType[*]", "field Struct.ListField[*]"},
		"listField[1]":        {"type ListType[*]", "field Struct.ListField[*]"},
		"listTypedefField[0]": {"type ListTypedefType[*]", "field Struct.ListTypedefField[*]"},
		"listTypedefField[1]": {"type ListTypedefType[*]", "field Struct.ListTypedefField[*]"},
	})
}
