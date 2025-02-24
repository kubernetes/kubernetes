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
		MapField: map[string]map[string]string{
			"a": {"x": "X", "y": "Y"},
			"b": {"x": "X", "y": "Y"},
		},
		MapPtrField: map[string]map[string]*string{
			"a": {"x": ptr.To("X"), "y": ptr.To("Y")},
			"b": {"x": ptr.To("X"), "y": ptr.To("Y")},
		},
		MapTypedefField: map[string]MapType{
			"a": {"x": "X", "y": "Y"},
			"b": {"x": "X", "y": "Y"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		"mapField[a][x]":        {"field Struct.MapField[*][*]"},
		"mapField[a][y]":        {"field Struct.MapField[*][*]"},
		"mapField[b][x]":        {"field Struct.MapField[*][*]"},
		"mapField[b][y]":        {"field Struct.MapField[*][*]"},
		"mapPtrField[a][x]":     {"field Struct.MapPtrField[*][*]"},
		"mapPtrField[a][y]":     {"field Struct.MapPtrField[*][*]"},
		"mapPtrField[b][x]":     {"field Struct.MapPtrField[*][*]"},
		"mapPtrField[b][y]":     {"field Struct.MapPtrField[*][*]"},
		"mapTypedefField[a][x]": {"field Struct.MapTypedefField[*][*]"},
		"mapTypedefField[a][y]": {"field Struct.MapTypedefField[*][*]"},
		"mapTypedefField[b][x]": {"field Struct.MapTypedefField[*][*]"},
		"mapTypedefField[b][y]": {"field Struct.MapTypedefField[*][*]"},
	})
}
