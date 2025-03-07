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

package deep

import (
	"testing"

	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values.
	}).ExpectValid()

	st.Value(&Struct{
		ListField: []map[string]string{
			{"a": "A", "b": "B"},
			{"c": "C", "d": "D"},
		},
		ListPtrField: []map[string]*string{
			{"a": ptr.To("A"), "b": ptr.To("B")},
			{"c": ptr.To("C"), "d": ptr.To("D")},
		},
		ListTypedefField: []MapType{
			{"a": "A", "b": "B"},
			{"c": "C", "d": "D"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		"listField[0][a]":        {"field Struct.ListField[*][*]"},
		"listField[0][b]":        {"field Struct.ListField[*][*]"},
		"listField[1][c]":        {"field Struct.ListField[*][*]"},
		"listField[1][d]":        {"field Struct.ListField[*][*]"},
		"listPtrField[0][a]":     {"field Struct.ListPtrField[*][*]"},
		"listPtrField[0][b]":     {"field Struct.ListPtrField[*][*]"},
		"listPtrField[1][c]":     {"field Struct.ListPtrField[*][*]"},
		"listPtrField[1][d]":     {"field Struct.ListPtrField[*][*]"},
		"listTypedefField[0][a]": {"field Struct.ListTypedefField[*][*]"},
		"listTypedefField[0][b]": {"field Struct.ListTypedefField[*][*]"},
		"listTypedefField[1][c]": {"field Struct.ListTypedefField[*][*]"},
		"listTypedefField[1][d]": {"field Struct.ListTypedefField[*][*]"},
	})
}
