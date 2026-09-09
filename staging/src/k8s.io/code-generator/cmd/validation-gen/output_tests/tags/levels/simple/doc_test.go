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

package simple

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestAlpha(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		IntField:           10,
		StringField:        "abc",
		SliceField:         []string{"a", "b"},
		UUIDField:          "a0a2a2d2-0b87-4964-a123-78d00a8787a6",
		ImmutableField:     "foo",
		IntFieldBeta:       10,
		StringFieldBeta:    "abc",
		SliceFieldBeta:     []string{"a", "b"},
		UUIDFieldBeta:      "a0a2a2d2-0b87-4964-a123-78d00a8787a6",
		ImmutableFieldBeta: "foo",
	}).ExpectValid()

	// Test failures marked as alpha
	st.Value(&Struct{
		IntField:           5,
		StringField:        "too-long",
		SliceField:         []string{"a", "b", "c"},
		UUIDField:          "not-a-uuid",
		ImmutableField:     "bar",
		IntFieldBeta:       10,
		StringFieldBeta:    "abc",
		SliceFieldBeta:     []string{"a", "b"},
		UUIDFieldBeta:      "a0a2a2d2-0b87-4964-a123-78d00a8787a6",
		ImmutableFieldBeta: "foo",
	}).OldValue(&Struct{
		ImmutableField:     "foo",
		ImmutableFieldBeta: "foo",
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), 5, "").WithOrigin("minimum").MarkAlpha(),
		field.TooLong(field.NewPath("stringField"), "too-long", 5).WithOrigin("maxLength").MarkAlpha(),
		field.TooMany(field.NewPath("sliceField"), 3, 2).WithOrigin("maxItems").MarkAlpha(),
		field.Invalid(field.NewPath("uuidField"), "not-a-uuid", "").WithOrigin("format=k8s-uuid").MarkAlpha(),
		field.Invalid(field.NewPath("immutableField"), "bar", "").WithOrigin("immutable").MarkAlpha(),
	})
}

func TestBeta(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		IntField:           10,
		StringField:        "abc",
		SliceField:         []string{"a", "b"},
		UUIDField:          "a0a2a2d2-0b87-4964-a123-78d00a8787a6",
		ImmutableField:     "foo",
		IntFieldBeta:       10,
		StringFieldBeta:    "abc",
		SliceFieldBeta:     []string{"a", "b"},
		UUIDFieldBeta:      "a0a2a2d2-0b87-4964-a123-78d00a8787a6",
		ImmutableFieldBeta: "foo",
	}).ExpectValid()

	// Test failures marked as beta
	st.Value(&Struct{
		IntField:           10,
		StringField:        "abc",
		SliceField:         []string{"a", "b"},
		UUIDField:          "a0a2a2d2-0b87-4964-a123-78d00a8787a6",
		ImmutableField:     "foo",
		IntFieldBeta:       5,
		StringFieldBeta:    "too-long",
		SliceFieldBeta:     []string{"a", "b", "c"},
		UUIDFieldBeta:      "not-a-uuid",
		ImmutableFieldBeta: "bar",
	}).OldValue(&Struct{
		ImmutableField:     "foo",
		ImmutableFieldBeta: "foo",
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(field.NewPath("intFieldBeta"), 5, "").WithOrigin("minimum").MarkBeta(),
		field.TooLong(field.NewPath("stringFieldBeta"), "too-long", 5).WithOrigin("maxLength").MarkBeta(),
		field.TooMany(field.NewPath("sliceFieldBeta"), 3, 2).WithOrigin("maxItems").MarkBeta(),
		field.Invalid(field.NewPath("uuidFieldBeta"), "not-a-uuid", "").WithOrigin("format=k8s-uuid").MarkBeta(),
		field.Invalid(field.NewPath("immutableFieldBeta"), "bar", "").WithOrigin("immutable").MarkBeta(),
	})
}

func TestSpecialValidationStruct(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&SpecialValidationStruct{
		NEQField:     10,
		NEQFieldBeta: 10,
	}).ExpectValid()

	st.Value(&SpecialValidationStruct{
		NEQField:           5,
		NEQFieldBeta:       5,
		ForbiddenField:     ptr.To("val"),
		ForbiddenFieldBeta: ptr.To("val"),
		UpdateField:        "new",
		UpdateFieldBeta:    "new",
	}).OldValue(&SpecialValidationStruct{
		UpdateField:     "old",
		UpdateFieldBeta: "old",
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(field.NewPath("neqField"), 5, "must not be 5").WithOrigin("neq").MarkAlpha(),
		field.Invalid(field.NewPath("neqFieldBeta"), 5, "must not be 5").WithOrigin("neq").MarkBeta(),
		field.Forbidden(field.NewPath("forbiddenField"), "").MarkAlpha(),
		field.Forbidden(field.NewPath("forbiddenFieldBeta"), "").MarkBeta(),
		field.Invalid(field.NewPath("updateField"), "new", "field cannot be modified once set").WithOrigin("update").MarkAlpha(),
		field.Invalid(field.NewPath("updateFieldBeta"), "new", "field cannot be modified once set").WithOrigin("update").MarkBeta(),
	})

	st.Value(&StructWithValidateFalse{
		ValidateFalse:     ptr.To("val"),
		ValidateFalseBeta: ptr.To("val"),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(field.NewPath("validateFalse"), "val", "always fails").MarkAlpha(),
		field.Invalid(field.NewPath("validateFalseBeta"), "val", "always fails").MarkBeta(),
	})
}
