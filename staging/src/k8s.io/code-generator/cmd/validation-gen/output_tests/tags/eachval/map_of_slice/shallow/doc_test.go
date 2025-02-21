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
package shallow

import (
	"testing"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values.
	}).ExpectValid()

	st.Value(&Struct{
		MapField:        map[string][]string{"a": make([]string, 0), "b": make([]string, 0)},
		MapPtrField:     map[string][]*string{"a": make([]*string, 0), "b": make([]*string, 0)},
		MapTypedefField: map[string]SliceType{"a": make(SliceType, 0), "b": make(SliceType, 0)},
	}).ExpectValidateFalseByPath(map[string][]string{
		"mapField[a]":        []string{"field Struct.MapField[*]"},
		"mapField[b]":        []string{"field Struct.MapField[*]"},
		"mapPtrField[a]":     []string{"field Struct.MapPtrField[*]"},
		"mapPtrField[b]":     []string{"field Struct.MapPtrField[*]"},
		"mapTypedefField[a]": []string{"field Struct.MapTypedefField[*]"},
		"mapTypedefField[b]": []string{"field Struct.MapTypedefField[*]"},
	})
}
