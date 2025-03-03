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

package typedeftomap

import (
	"testing"

	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values.
	}).ExpectValidateFalseByPath(map[string][]string{
		"": {"type Struct"},
	})

	st.Value(&Struct{
		MapField:        MapType{"a": "A", "b": "B"},
		MapPtrField:     MapPtrType{"a": ptr.To("A"), "b": ptr.To("B")},
		MapTypedefField: MapTypedefType{"a": StringType("A"), "b": StringType("B")},
	}).ExpectValidateFalseByPath(map[string][]string{
		"":                   {"type Struct"},
		"mapField[a]":        {"type MapType[*]", "field Struct.MapField[*]"},
		"mapField[b]":        {"type MapType[*]", "field Struct.MapField[*]"},
		"mapPtrField[a]":     {"type MapPtrType[*]", "field Struct.MapPtrField[*]"},
		"mapPtrField[b]":     {"type MapPtrType[*]", "field Struct.MapPtrField[*]"},
		"mapTypedefField[a]": {"type MapTypedefType[*]", "field Struct.MapTypedefField[*]", "type StringType"},
		"mapTypedefField[b]": {"type MapTypedefType[*]", "field Struct.MapTypedefField[*]", "type StringType"},
	})
}
