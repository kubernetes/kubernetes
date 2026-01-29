/*
Copyright 2025 The Kubernetes Authors.

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

package fields

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestShadow(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		IntField:       10,
		StringField:    "abc",
		SliceField:     []string{"a", "b"},
		UUIDField:      "a0a2a2d2-0b87-4964-a123-78d00a8787a6",
		Subfield:       SubStruct{Inner: 5},
		RequiredField:  ptr.To("val"),
		EnumField:      "A",
		D:              DM1,
		M1:             &M1{},
		ImmutableField: "foo",
	}).ExpectValid()

	// Test failures marked as shadow
	st.Value(&Struct{
		IntField:       5,
		StringField:    "too-long",
		SliceField:     []string{"a", "b", "c"},
		UUIDField:      "not-a-uuid",
		Subfield:       SubStruct{Inner: 1},
		RequiredField:  nil,
		EnumField:      "B",
		D:              DM1,
		M1:             nil, // required by discriminator
		ImmutableField: "bar",
	}).OldValue(&Struct{
		RequiredField:  ptr.To("old"),
		ImmutableField: "foo",
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByShadow(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), 5, "").WithOrigin("minimum").MarkShadow(),
		field.TooLong(field.NewPath("stringField"), "too-long", 5).WithOrigin("maxLength").MarkShadow(),
		field.TooMany(field.NewPath("sliceField"), 3, 2).WithOrigin("maxItems").MarkShadow(),
		field.Invalid(field.NewPath("uuidField"), "not-a-uuid", "").WithOrigin("format=k8s-uuid").MarkShadow(),
		field.Invalid(field.NewPath("subfield", "inner"), 1, "").WithOrigin("minimum").MarkShadow(),
		field.Required(field.NewPath("requiredField"), "").MarkShadow(),
		field.NotSupported(field.NewPath("enumField"), Enum("B"), []string{"A"}).MarkShadow(),
		field.Invalid(field.NewPath("m1"), nil, "").WithOrigin("union").MarkShadow(),
		field.Invalid(field.NewPath("immutableField"), "bar", "").WithOrigin("immutable").MarkShadow(),
	})
}

func TestMixed(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid case (meets both normal and shadow requirements)
	st.Value(&MixedStruct{
		IntField:  15,
		ListField: []string{"a", "b", "c"},
	}).ExpectValid()

	// Fails shadow validation but passes normal validation
	// IntField: 5 <= 8 < 10 (shadow fails)
	// ListField: 3 < 4 <= 5 (shadow fails)
	st.Value(&MixedStruct{
		IntField:  8,
		ListField: []string{"a", "b", "c", "d"},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByShadow(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), 8, "").WithOrigin("minimum").MarkShadow(),
		field.TooMany(field.NewPath("listField"), 4, 3).WithOrigin("maxItems").MarkShadow(),
	})

	// Fails both normal and shadow validation
	// IntField: 4 < 5 (normal fails) AND 4 < 10 (shadow fails)
	// ListField: 6 > 5 (normal fails) AND 6 > 3 (shadow fails)
	st.Value(&MixedStruct{
		IntField:  4,
		ListField: []string{"a", "b", "c", "d", "e", "f"},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByShadow(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), 4, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("intField"), 4, "").WithOrigin("minimum").MarkShadow(),
		field.TooMany(field.NewPath("listField"), 6, 5).WithOrigin("maxItems"),
		field.TooMany(field.NewPath("listField"), 6, 3).WithOrigin("maxItems").MarkShadow(),
	})
}

func TestMyStruct(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&MyStruct{
		NEQField:         10,
		ConditionalField: 15,
		RecursiveShadow:  25,
	}).Opts([]string{"MyFeature"}).ExpectValid()

	st.Value(&MyStruct{
		NEQField:         5,
		ForbiddenField:   ptr.To("val"),
		UpdateField:      "new",
		Z1:               &Z1{},
		Z2:               &Z2{},
		ConditionalField: 5,
		RecursiveShadow:  10,
	}).Opts([]string{"MyFeature"}).OldValue(&MyStruct{
		UpdateField: "old",
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByShadow(), field.ErrorList{
		field.Invalid(field.NewPath("neqField"), 5, "must not be 5").WithOrigin("neq").MarkShadow(),
		field.Forbidden(field.NewPath("forbiddenField"), "").MarkShadow(),
		field.Invalid(field.NewPath("updateField"), "new", "field cannot be modified once set").WithOrigin("update").MarkShadow(),
		field.Invalid(nil, &MyStruct{NEQField: 5, ForbiddenField: ptr.To("val"), UpdateField: "new", Z1: &Z1{}, Z2: &Z2{}, ConditionalField: 5, RecursiveShadow: 10}, "only one of z1, z2 may be specified").WithOrigin("zeroOrOneOf").MarkShadow(),
		field.Invalid(field.NewPath("conditionalField"), 5, "").WithOrigin("minimum").MarkShadow(),
		field.Invalid(field.NewPath("recursiveShadow"), 10, "").WithOrigin("minimum").MarkShadow(),
	})

	st.Value(&StructWithValidateFalse{
		ValidateFalse: ptr.To("val"),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByShadow(), field.ErrorList{
		field.Invalid(field.NewPath("validateFalse"), "val", "always fails").MarkShadow(),
	})
}
