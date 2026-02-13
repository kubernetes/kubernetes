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

package discriminator

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestStrictUnion(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Mode A: FieldA required, FieldB implicitly forbidden
	st.Value(&StrictUnion{D1: "A", FieldA: ptr.To("val")}).ExpectValid()
	st.Value(&StrictUnion{D1: "A"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("fieldA"), ""),
	})
	st.Value(&StrictUnion{D1: "A", FieldA: ptr.To("val"), FieldB: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("fieldB"), ""),
	})

	// Mode B: FieldA implicitly forbidden, FieldB required
	st.Value(&StrictUnion{D1: "B", FieldB: ptr.To("val")}).ExpectValid()
	st.Value(&StrictUnion{D1: "B"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("fieldB"), ""),
	})
	st.Value(&StrictUnion{D1: "B", FieldA: ptr.To("val"), FieldB: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("fieldA"), ""),
	})
}

func TestSharedField(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid (optional) in A and B
	st.Value(&SharedField{D1: "A"}).ExpectValid()
	st.Value(&SharedField{D1: "A", FieldA: ptr.To("val")}).ExpectValid()
	st.Value(&SharedField{D1: "B"}).ExpectValid()
	st.Value(&SharedField{D1: "B", FieldA: ptr.To("val")}).ExpectValid()

	// Forbidden in C
	st.Value(&SharedField{D1: "C", FieldA: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("fieldA"), ""),
	})
}

func TestChainedValidation(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Mode A: Required AND maxLength 5
	st.Value(&ChainedValidation{D1: "A", FieldA: ptr.To("abc")}).ExpectValid()
	st.Value(&ChainedValidation{D1: "A"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("fieldA"), ""),
	})
	st.Value(&ChainedValidation{D1: "A", FieldA: ptr.To("too-long")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooLong(field.NewPath("fieldA"), "too-long", 5),
	})

	// Mode B: Unlisted, so implicitly forbidden
	st.Value(&ChainedValidation{D1: "B", FieldA: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("fieldA"), ""),
	})
}

func TestImplicitForbidden(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Mode A: Optional
	st.Value(&ImplicitForbidden{D1: "A"}).ExpectValid()
	st.Value(&ImplicitForbidden{D1: "A", FieldA: ptr.To("val")}).ExpectValid()

	// Mode B: Not listed, so implicitly Forbidden
	st.Value(&ImplicitForbidden{D1: "B", FieldA: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("fieldA"), ""),
	})
}

func TestNonStringDiscriminator(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Bool mode
	st.Value(&NonStringDiscriminator{D1: true, FieldA: ptr.To("val")}).ExpectValid()
	st.Value(&NonStringDiscriminator{D1: true}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("fieldA"), ""),
	})
	st.Value(&NonStringDiscriminator{D1: false, FieldA: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("fieldA"), ""),
	})

	// Int mode
	st.Value(&NonStringDiscriminator{D2: 1, FieldB: ptr.To("val")}).ExpectValid()
	st.Value(&NonStringDiscriminator{D2: 1}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("fieldB"), ""),
	})
	st.Value(&NonStringDiscriminator{D2: 2, FieldB: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("fieldB"), ""),
	})
}

func TestMultipleDiscriminators(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&MultipleDiscriminators{
		D1:     "A",
		D2:     "B",
		FieldA: ptr.To("valA"),
		FieldB: ptr.To("valB"),
	}).ExpectValid()

	st.Value(&MultipleDiscriminators{
		D1: "A",
		D2: "B",
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("fieldA"), ""),
		field.Required(field.NewPath("fieldB"), ""),
	})
}

func TestCollections(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Mode A: Collections are valid (optional)
	st.Value(&Collections{
		D1: "A",
	}).ExpectValid()

	st.Value(&Collections{
		D1:        "A",
		ListField: []string{"item"},
		MapField:  map[string]string{"key": "val"},
	}).ExpectValid()

	// Mode B: Unlisted, so implicitly forbidden
	st.Value(&Collections{
		D1:        "B",
		ListField: []string{"item"},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("listField"), ""),
	})

	st.Value(&Collections{
		D1:       "B",
		MapField: map[string]string{"key": "val"},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("mapField"), ""),
	})
}

func TestRatcheting(t *testing.T) {
	mkTest := func() *ChainedValidation {
		return &ChainedValidation{
			D1:     "A",
			FieldA: ptr.To("too-long-string"),
		}
	}

	st := localSchemeBuilder.Test(t)

	// 1. New object is invalid
	st.Value(mkTest()).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooLong(field.NewPath("fieldA"), "too-long-string", 5),
	})

	// 2. Unchanged update is valid (ratcheting)
	st.Value(mkTest()).OldValue(mkTest()).ExpectValid()

	// 3. Changed value re-validates (and fails)
	mkDifferent := func() *ChainedValidation {
		return &ChainedValidation{
			D1:     "A",
			FieldA: ptr.To("also-too-long"),
		}
	}
	st.Value(mkTest()).OldValue(mkDifferent()).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooLong(field.NewPath("fieldA"), "too-long-string", 5),
	})

	// 4. Changed discriminator re-validates (and fails)
	mkDifferentDisc := func() *ChainedValidation {
		return &ChainedValidation{
			D1:     "B", // Discriminator changed from B -> A
			FieldA: ptr.To("too-long-string"),
		}
	}
	st.Value(mkTest()).OldValue(mkDifferentDisc()).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooLong(field.NewPath("fieldA"), "too-long-string", 5),
	})
}
