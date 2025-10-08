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

package required

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{ /* All zero-values */ }).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("stringField"), ""),
		field.Required(field.NewPath("stringPtrField"), ""),
		field.Required(field.NewPath("stringTypedefField"), ""),
		field.Required(field.NewPath("stringTypedefPtrField"), ""),
		field.Required(field.NewPath("intField"), ""),
		field.Required(field.NewPath("intPtrField"), ""),
		field.Required(field.NewPath("intTypedefField"), ""),
		field.Required(field.NewPath("intTypedefPtrField"), ""),
		field.Required(field.NewPath("otherStructPtrField"), ""),
		field.Required(field.NewPath("sliceField"), ""),
		field.Required(field.NewPath("sliceTypedefField"), ""),
		field.Required(field.NewPath("mapField"), ""),
		field.Required(field.NewPath("mapTypedefField"), ""),
	})

	// Test validation ratcheting
	st.Value(&Struct{}).OldValue(&Struct{}).ExpectValid()

	st.Value(&Struct{
		StringPtrField:        ptr.To(""),             // satisfies required
		StringTypedefPtrField: ptr.To(StringType("")), // satisfies required
		IntPtrField:           ptr.To(0),              // satisfies required
		IntTypedefPtrField:    ptr.To(IntType(0)),     // satisfies required
		SliceField:            []string{},             // does not satisfy required
		SliceTypedefField:     []string{},             // does not satisfy required
		MapField:              map[string]string{},    // does not satisfy required
		MapTypedefField:       map[string]string{},    // does not satisfy required
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring(), field.ErrorList{
		field.Required(field.NewPath("stringField"), ""),
		field.Invalid(field.NewPath("stringPtrField"), ptr.To(""), "field Struct.StringPtrField"),
		field.Required(field.NewPath("stringTypedefField"), ""),
		field.Invalid(field.NewPath("stringTypedefPtrField"), ptr.To(StringType("")), "field Struct.StringTypedefPtrField"),
		field.Invalid(field.NewPath("stringTypedefPtrField"), ptr.To(StringType("")), "type StringType"),
		field.Required(field.NewPath("intField"), ""),
		field.Invalid(field.NewPath("intPtrField"), ptr.To(0), "field Struct.IntPtrField"),
		field.Required(field.NewPath("intTypedefField"), ""),
		field.Invalid(field.NewPath("intTypedefPtrField"), ptr.To(IntType(0)), "field Struct.IntTypedefPtrField"),
		field.Invalid(field.NewPath("intTypedefPtrField"), ptr.To(IntType(0)), "type IntType"),
		field.Required(field.NewPath("otherStructPtrField"), ""),
		field.Required(field.NewPath("sliceField"), ""),
		field.Required(field.NewPath("sliceTypedefField"), ""),
		field.Required(field.NewPath("mapField"), ""),
		field.Required(field.NewPath("mapTypedefField"), ""),
	})
	// Test validation ratcheting
	st.Value(&Struct{
		StringPtrField:        ptr.To(""),             // satisfies required
		StringTypedefPtrField: ptr.To(StringType("")), // satisfies required
		IntPtrField:           ptr.To(0),              // satisfies required
		IntTypedefPtrField:    ptr.To(IntType(0)),     // satisfies required
		SliceField:            []string{},             // does not satisfy required
		SliceTypedefField:     []string{},             // does not satisfy required
		MapField:              map[string]string{},    // does not satisfy required
		MapTypedefField:       map[string]string{},    // does not satisfy required
	}).OldValue(&Struct{
		StringPtrField:        ptr.To(""),             // satisfies required
		StringTypedefPtrField: ptr.To(StringType("")), // satisfies required
		IntPtrField:           ptr.To(0),              // satisfies required
		IntTypedefPtrField:    ptr.To(IntType(0)),     // satisfies required
		// nil and empty slices are considered equivalent.
	}).ExpectValid()

	mkInvalid := func() *Struct {
		return &Struct{
			StringField:           "abc",
			StringPtrField:        ptr.To("xyz"),
			StringTypedefField:    StringType("abc"),
			StringTypedefPtrField: ptr.To(StringType("xyz")),
			IntField:              123,
			IntPtrField:           ptr.To(456),
			IntTypedefField:       IntType(123),
			IntTypedefPtrField:    ptr.To(IntType(456)),
			OtherStructPtrField:   &OtherStruct{},
			SliceField:            []string{"a", "b"},
			SliceTypedefField:     SliceType([]string{"a", "b"}),
			MapField:              map[string]string{"a": "b", "c": "d"},
			MapTypedefField:       MapType(map[string]string{"a": "b", "c": "d"}),
		}
	}

	st.Value(mkInvalid()).ExpectValidateFalseByPath(map[string][]string{
		"stringField":           {"field Struct.StringField"},
		"stringPtrField":        {"field Struct.StringPtrField"},
		"stringTypedefField":    {"field Struct.StringTypedefField", "type StringType"},
		"stringTypedefPtrField": {"field Struct.StringTypedefPtrField", "type StringType"},
		"intField":              {"field Struct.IntField"},
		"intPtrField":           {"field Struct.IntPtrField"},
		"intTypedefField":       {"field Struct.IntTypedefField", "type IntType"},
		"intTypedefPtrField":    {"field Struct.IntTypedefPtrField", "type IntType"},
		"otherStructPtrField":   {"field Struct.OtherStructPtrField", "type OtherStruct"},
		"sliceField":            {"field Struct.SliceField"},
		"sliceTypedefField":     {"field Struct.SliceTypedefField", "type SliceType"},
		"mapField":              {"field Struct.MapField"},
		"mapTypedefField":       {"field Struct.MapTypedefField", "type MapType"},
	})
	// Test validation ratcheting
	st.Value(mkInvalid()).OldValue(mkInvalid()).ExpectValid()
}
