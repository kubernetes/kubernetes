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

package modes

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestAlpha(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid: mode A with FieldA set
	st.Value(&AlphaStruct{D1: "A", FieldA: ptr.To("val")}).ExpectValid()

	// Invalid: mode A with FieldA missing (required), should be stability level alpha
	st.Value(&AlphaStruct{D1: "A"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("fieldA"), "").MarkAlpha(),
	})

	// Invalid: mode A with FieldB set (forbidden), should be stability level alpha
	st.Value(&AlphaStruct{D1: "A", FieldA: ptr.To("val"), FieldB: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Forbidden(field.NewPath("fieldB"), "").MarkAlpha(),
	})

	// Valid: mode B with FieldB set
	st.Value(&AlphaStruct{D1: "B", FieldB: ptr.To("val")}).ExpectValid()

	// Invalid: mode B with FieldB missing (required), should be stability level alpha
	st.Value(&AlphaStruct{D1: "B"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("fieldB"), "").MarkAlpha(),
	})
}

func TestBeta(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid: mode A with FieldA set
	st.Value(&BetaStruct{D1: "A", FieldA: ptr.To("val")}).ExpectValid()

	// Invalid: mode A with FieldA missing (required), should be stability level beta
	st.Value(&BetaStruct{D1: "A"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("fieldA"), "").MarkBeta(),
	})

	// Invalid: mode A with FieldB set (forbidden), should be stability level beta
	st.Value(&BetaStruct{D1: "A", FieldA: ptr.To("val"), FieldB: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Forbidden(field.NewPath("fieldB"), "").MarkBeta(),
	})

	// Valid: mode B with FieldB set
	st.Value(&BetaStruct{D1: "B", FieldB: ptr.To("val")}).ExpectValid()

	// Invalid: mode B with FieldB missing (required), should be stability level beta
	st.Value(&BetaStruct{D1: "B"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("fieldB"), "").MarkBeta(),
	})
}

func TestMixedLevels(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid cases
	st.Value(&MixedLevels{Mode: "A", A: ptr.To("val")}).ExpectValid()
	st.Value(&MixedLevels{Mode: "B", B: ptr.To("val")}).ExpectValid()

	// Mode=A, missing A -> alpha error (field A is alpha-gated)
	st.Value(&MixedLevels{Mode: "A"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("a"), "").MarkAlpha(),
	})

	// Mode=B, missing B -> beta error (field B is beta-gated)
	st.Value(&MixedLevels{Mode: "B"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("b"), "").MarkBeta(),
	})

	// Mode=A with B set (forbidden) -> beta error (field B's +k8s:modeDiscriminator is beta)
	st.Value(&MixedLevels{Mode: "A", A: ptr.To("val"), B: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Forbidden(field.NewPath("b"), "").MarkBeta(),
	})

	// Mode=B with A set (forbidden) -> alpha error (field A's +k8s:modeDiscriminator is alpha)
	st.Value(&MixedLevels{Mode: "B", A: ptr.To("val"), B: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Forbidden(field.NewPath("a"), "").MarkAlpha(),
	})
}

func TestCrossLevels(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid cases
	st.Value(&CrossLevels{Kind: "A", A: ptr.To("val")}).ExpectValid()
	st.Value(&CrossLevels{Kind: "B", B: ptr.To("val")}).ExpectValid()

	// Kind=A, missing A -> alpha error (field A is alpha-gated, discriminator is beta)
	st.Value(&CrossLevels{Kind: "A"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("a"), "").MarkAlpha(),
	})

	// Kind=B, missing B -> alpha error (field B is alpha-gated, discriminator is beta)
	st.Value(&CrossLevels{Kind: "B"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("b"), "").MarkAlpha(),
	})

	// Kind=A with B set (forbidden) -> alpha error (field B's member tag is alpha)
	st.Value(&CrossLevels{Kind: "A", A: ptr.To("val"), B: ptr.To("val")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Forbidden(field.NewPath("b"), "").MarkAlpha(),
	})
}

func TestSameFieldMixed(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid cases
	st.Value(&SameFieldMixed{Mode: "A", Value: ptr.To("val")}).ExpectValid()
	st.Value(&SameFieldMixed{Mode: "B", Value: ptr.To("val")}).ExpectValid()

	// Mode=A, missing Value -> alpha error (member("A") is alpha-gated)
	st.Value(&SameFieldMixed{Mode: "A"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("value"), "").MarkAlpha(),
	})

	// Mode=B, missing Value -> beta error (member("B") is beta-gated)
	st.Value(&SameFieldMixed{Mode: "B"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("value"), "").MarkBeta(),
	})
}

// TestSameValueMixedPayloads verifies that multiple payload validations on
// the same discriminator value can have different stability levels.
func TestSameValueMixedPayloads(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid: Mode=A with Value set and long enough
	st.Value(&SameValueMixedPayloads{Mode: "A", Value: ptr.To("abc")}).ExpectValid()

	// Mode=A, missing Value -> alpha error (required is alpha-gated)
	st.Value(&SameValueMixedPayloads{Mode: "A"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("value"), "").MarkAlpha(),
	})

	// Mode=A, Value too short -> beta error (minLength is beta-gated)
	st.Value(&SameValueMixedPayloads{Mode: "A", Value: ptr.To("ab")}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByValidationStabilityLevel(), field.ErrorList{
		field.TooShort(field.NewPath("value"), "", 3).MarkBeta(),
	})
}
