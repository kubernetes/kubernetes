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

package discriminators

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestAlpha(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid: mode A with FieldA set
	st.Value(&AlphaStruct{D1: "A", FieldA: ptr.To("val")}).ExpectValid()

	// Invalid: mode A with FieldA missing (required) - should be alpha
	st.Value(&AlphaStruct{D1: "A"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("fieldA"), "").MarkAlpha(),
	})

	// Invalid: mode A with FieldB set (forbidden) - should be alpha
	st.Value(&AlphaStruct{D1: "A", FieldA: ptr.To("val"), FieldB: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Forbidden(field.NewPath("fieldB"), "").MarkAlpha(),
	})

	// Valid: mode B with FieldB set
	st.Value(&AlphaStruct{D1: "B", FieldB: ptr.To("val")}).ExpectValid()

	// Invalid: mode B with FieldB missing (required) - should be alpha
	st.Value(&AlphaStruct{D1: "B"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("fieldB"), "").MarkAlpha(),
	})
}

func TestBeta(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid: mode A with FieldA set
	st.Value(&BetaStruct{D1: "A", FieldA: ptr.To("val")}).ExpectValid()

	// Invalid: mode A with FieldA missing (required) - should be beta
	st.Value(&BetaStruct{D1: "A"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("fieldA"), "").MarkBeta(),
	})

	// Invalid: mode A with FieldB set (forbidden) - should be beta
	st.Value(&BetaStruct{D1: "A", FieldA: ptr.To("val"), FieldB: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Forbidden(field.NewPath("fieldB"), "").MarkBeta(),
	})

	// Valid: mode B with FieldB set
	st.Value(&BetaStruct{D1: "B", FieldB: ptr.To("val")}).ExpectValid()

	// Invalid: mode B with FieldB missing (required) - should be beta
	st.Value(&BetaStruct{D1: "B"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("fieldB"), "").MarkBeta(),
	})
}
