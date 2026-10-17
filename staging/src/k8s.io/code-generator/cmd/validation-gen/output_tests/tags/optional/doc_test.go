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

package optional

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero-values.
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("otherStructField"), nil, ""),
	})

	st.Value(&Struct{
		StringField:           "abc",
		StringPtrField:        ptr.To("xyz"),
		StringTypedefField:    StringType("abc"),
		StringTypedefPtrField: ptr.To(StringType("xyz")),
		IntField:              123,
		IntPtrField:           ptr.To(456),
		IntTypedefField:       IntType(123),
		IntTypedefPtrField:    ptr.To(IntType(456)),
		OtherStructField:      OtherStruct{},
		OtherStructPtrField:   &OtherStruct{},
		SliceField:            []string{"a", "b"},
		SliceTypedefField:     SliceType([]string{"a", "b"}),
		MapField:              map[string]string{"a": "b", "c": "d"},
		MapTypedefField:       MapType(map[string]string{"a": "b", "c": "d"}),
	}).ExpectValidateFalseByPath(map[string][]string{
		"stringField":           {"field Struct.StringField"},
		"stringPtrField":        {"field Struct.StringPtrField"},
		"stringTypedefField":    {"field Struct.StringTypedefField", "type StringType"},
		"stringTypedefPtrField": {"field Struct.StringTypedefPtrField", "type StringType"},
		"intField":              {"field Struct.IntField"},
		"intPtrField":           {"field Struct.IntPtrField"},
		"intTypedefField":       {"field Struct.IntTypedefField", "type IntType"},
		"intTypedefPtrField":    {"field Struct.IntTypedefPtrField", "type IntType"},
		"otherStructField":      {"type OtherStruct"},
		"otherStructPtrField":   {"type OtherStruct", "field Struct.OtherStructPtrField"},
		"sliceField":            {"field Struct.SliceField"},
		"sliceTypedefField":     {"field Struct.SliceTypedefField", "type SliceType"},
		"mapField":              {"field Struct.MapField"},
		"mapTypedefField":       {"field Struct.MapTypedefField", "type MapType"},
	})
}
