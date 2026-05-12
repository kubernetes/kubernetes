/*
Copyright The Kubernetes Authors.

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

package required

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero-values (nil slices/maps).
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("stringField"), ""),
		field.Required(field.NewPath("stringPtrField"), ""),
		field.Required(field.NewPath("stringTypedefField"), ""),
		field.Required(field.NewPath("stringTypedefPtrField"), ""),
		field.Required(field.NewPath("intField"), ""),
		field.Required(field.NewPath("intPtrField"), ""),
		field.Required(field.NewPath("intTypedefField"), ""),
		field.Required(field.NewPath("intTypedefPtrField"), ""),
		field.Required(field.NewPath("boolField"), ""),
		field.Required(field.NewPath("floatField"), ""),
		field.Required(field.NewPath("byteField"), ""),
		field.Required(field.NewPath("otherStructPtrField"), ""),
		field.Required(field.NewPath("sliceField"), ""),
		field.Required(field.NewPath("sliceTypedefField"), ""),
		field.Required(field.NewPath("byteArrayField"), ""),
		field.Required(field.NewPath("mapField"), ""),
		field.Required(field.NewPath("mapTypedefField"), ""),
	})

	st.Value(&Struct{
		// Explicit zero-values and empty slices/maps.
		StringField:           "",
		StringPtrField:        nil,
		StringTypedefField:    "",
		StringTypedefPtrField: nil,
		IntField:              0,
		IntPtrField:           nil,
		IntTypedefField:       0,
		IntTypedefPtrField:    nil,
		BoolField:             false,
		FloatField:            0.0,
		ByteField:             0,
		OtherStructPtrField:   nil,
		SliceField:            []string{},
		SliceTypedefField:     SliceType{},
		ByteArrayField:        []byte{},
		MapField:              map[string]string{},
		MapTypedefField:       MapType{},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("stringField"), ""),
		field.Required(field.NewPath("stringPtrField"), ""),
		field.Required(field.NewPath("stringTypedefField"), ""),
		field.Required(field.NewPath("stringTypedefPtrField"), ""),
		field.Required(field.NewPath("intField"), ""),
		field.Required(field.NewPath("intPtrField"), ""),
		field.Required(field.NewPath("intTypedefField"), ""),
		field.Required(field.NewPath("intTypedefPtrField"), ""),
		field.Required(field.NewPath("boolField"), ""),
		field.Required(field.NewPath("floatField"), ""),
		field.Required(field.NewPath("byteField"), ""),
		field.Required(field.NewPath("otherStructPtrField"), ""),
		field.Required(field.NewPath("sliceField"), ""),
		field.Required(field.NewPath("sliceTypedefField"), ""),
		field.Required(field.NewPath("byteArrayField"), ""),
		field.Required(field.NewPath("mapField"), ""),
		field.Required(field.NewPath("mapTypedefField"), ""),
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
		BoolField:             true,
		FloatField:            1.23,
		ByteField:             'a',
		OtherStructPtrField:   &OtherStruct{},
		SliceField:            []string{"a", "b"},
		SliceTypedefField:     SliceType([]string{"a", "b"}),
		ByteArrayField:        []byte("abc"),
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
		"boolField":             {"field Struct.BoolField"},
		"floatField":            {"field Struct.FloatField"},
		"byteField":             {"field Struct.ByteField"},
		"otherStructPtrField":   {"type OtherStruct", "field Struct.OtherStructPtrField"},
		"sliceField":            {"field Struct.SliceField"},
		"sliceTypedefField":     {"field Struct.SliceTypedefField", "type SliceType"},
		"byteArrayField":        {"field Struct.ByteArrayField"},
		"mapField":              {"field Struct.MapField"},
		"mapTypedefField":       {"field Struct.MapTypedefField", "type MapType"},
	})
}
