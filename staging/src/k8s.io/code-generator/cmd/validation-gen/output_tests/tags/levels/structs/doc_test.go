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

package structs

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestMixed(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid case (meets both normal and alpha requirements)
	st.Value(&MixedStruct{
		IntField:      15,
		IntFieldBeta:  15,
		ListField:     []string{"a", "b", "c"},
		ListFieldBeta: []string{"a", "b", "c"},
	}).ExpectValid()

	// Fails alpha validation but passes normal validation
	// IntField: 5 <= 8 < 10 (alpha fails)
	// ListField: 3 < 4 <= 5 (alpha fails)
	st.Value(&MixedStruct{
		IntField:      8,
		IntFieldBeta:  8,
		ListField:     []string{"a", "b", "c", "d"},
		ListFieldBeta: []string{"a", "b", "c", "d"},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), 8, "").WithOrigin("minimum").MarkAlpha(),
		field.Invalid(field.NewPath("intFieldBeta"), 8, "").WithOrigin("minimum").MarkBeta(),
		field.TooMany(field.NewPath("listField"), 4, 3).WithOrigin("maxItems").MarkAlpha(),
		field.TooMany(field.NewPath("listFieldBeta"), 4, 3).WithOrigin("maxItems").MarkBeta(),
	})

	// Fails both normal and alpha validation
	// IntField: 4 < 5 (normal fails) AND 4 < 10 (alpha fails)
	// ListField: 6 > 5 (normal fails) AND 6 > 3 (alpha fails)
	st.Value(&MixedStruct{
		IntField:      4,
		IntFieldBeta:  4,
		ListField:     []string{"a", "b", "c", "d", "e", "f"},
		ListFieldBeta: []string{"a", "b", "c", "d", "e", "f"},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), 4, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("intField"), 4, "").WithOrigin("minimum").MarkAlpha(),
		field.Invalid(field.NewPath("intFieldBeta"), 4, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("intFieldBeta"), 4, "").WithOrigin("minimum").MarkBeta(),
		field.TooMany(field.NewPath("listField"), 6, 5).WithOrigin("maxItems"),
		field.TooMany(field.NewPath("listField"), 6, 3).WithOrigin("maxItems").MarkAlpha(),
		field.TooMany(field.NewPath("listFieldBeta"), 6, 5).WithOrigin("maxItems"),
		field.TooMany(field.NewPath("listFieldBeta"), 6, 3).WithOrigin("maxItems").MarkBeta(),
	})
}

func TestConditionalStruct(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&ConditionalStruct{
		ConditionalField:     15,
		ConditionalFieldBeta: 15,
		RecursiveAlpha:       25,
		RecursiveBeta:        25,
	}).Opts([]string{"MyFeature"}).ExpectValid()

	st.Value(&ConditionalStruct{
		ConditionalField:     5,
		ConditionalFieldBeta: 5,
		RecursiveAlpha:       10,
		RecursiveBeta:        10,
	}).Opts([]string{"MyFeature"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(field.NewPath("conditionalField"), 5, "").WithOrigin("minimum").MarkAlpha(),
		field.Invalid(field.NewPath("conditionalFieldBeta"), 5, "").WithOrigin("minimum").MarkBeta(),
		field.Invalid(field.NewPath("recursiveAlpha"), 10, "").WithOrigin("minimum").MarkAlpha(),
		field.Invalid(field.NewPath("recursiveBeta"), 10, "").WithOrigin("minimum").MarkBeta(),
	})
}
