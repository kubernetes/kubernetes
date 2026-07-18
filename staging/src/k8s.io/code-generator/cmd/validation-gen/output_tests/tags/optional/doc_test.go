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
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero-values.
	}).ExpectValid()

	st.Value(&Struct{
		StringField:           "abc",
		StringPtrField:        new("xyz"),
		StringTypedefField:    StringType("abc"),
		StringTypedefPtrField: new(StringType("xyz")),
		IntField:              123,
		IntPtrField:           new(456),
		IntTypedefField:       IntType(123),
		IntTypedefPtrField:    new(IntType(456)),
		OtherStructPtrField:   &OtherStruct{},
		SliceField:            []string{"a", "b"},
		SliceTypedefField:     SliceType([]string{"a", "b"}),
		MapField:              map[string]string{"a": "b", "c": "d"},
		MapTypedefField:       MapType(map[string]string{"a": "b", "c": "d"}),
		ChainedOptionalField:  new("short"),
		ChainedOptionalSubfield: NestedStruct{
			StructPtrField: &AnotherStruct{StringField: "s"},
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring(), field.ErrorList{
		field.Invalid(field.NewPath("stringField"), nil, "field Struct.StringField"),
		field.Invalid(field.NewPath("stringPtrField"), nil, "field Struct.StringPtrField"),
		field.Invalid(field.NewPath("stringTypedefField"), nil, "field Struct.StringTypedefField"),
		field.Invalid(field.NewPath("stringTypedefField"), nil, "type StringType"),
		field.Invalid(field.NewPath("stringTypedefPtrField"), nil, "field Struct.StringTypedefPtrField"),
		field.Invalid(field.NewPath("stringTypedefPtrField"), nil, "type StringType"),
		field.Invalid(field.NewPath("intField"), nil, "field Struct.IntField"),
		field.Invalid(field.NewPath("intPtrField"), nil, "field Struct.IntPtrField"),
		field.Invalid(field.NewPath("intTypedefField"), nil, "field Struct.IntTypedefField"),
		field.Invalid(field.NewPath("intTypedefField"), nil, "type IntType"),
		field.Invalid(field.NewPath("intTypedefPtrField"), nil, "field Struct.IntTypedefPtrField"),
		field.Invalid(field.NewPath("intTypedefPtrField"), nil, "type IntType"),
		field.Invalid(field.NewPath("otherStructPtrField"), nil, "type OtherStruct"),
		field.Invalid(field.NewPath("otherStructPtrField"), nil, "field Struct.OtherStructPtrField"),
		field.Invalid(field.NewPath("sliceField"), nil, "field Struct.SliceField"),
		field.Invalid(field.NewPath("sliceTypedefField"), nil, "field Struct.SliceTypedefField"),
		field.Invalid(field.NewPath("sliceTypedefField"), nil, "type SliceType"),
		field.Invalid(field.NewPath("mapField"), nil, "field Struct.MapField"),
		field.Invalid(field.NewPath("mapTypedefField"), nil, "field Struct.MapTypedefField"),
		field.Invalid(field.NewPath("mapTypedefField"), nil, "type MapType"),
		field.Invalid(field.NewPath("chainedOptionalField"), nil, "field Struct.ChainedOptionalField"),
		field.Invalid(field.NewPath("chainedOptionalSubfield.structPtrField.stringField"), nil, "field Struct.ChainedOptionalSubfield.StructPtrField.StringField"),
	})
}
