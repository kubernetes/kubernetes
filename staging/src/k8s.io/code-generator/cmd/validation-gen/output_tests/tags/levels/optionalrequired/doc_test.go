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

package optionalrequired

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestAlpha(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		RequiredField:     ptr.To("val"),
		RequiredFieldBeta: ptr.To("val"),
	}).ExpectValid()

	// Test failures marked as alpha
	st.Value(&Struct{
		RequiredField:     nil,
		RequiredFieldBeta: ptr.To("val"),
	}).OldValue(&Struct{
		RequiredField:     ptr.To("old"),
		RequiredFieldBeta: ptr.To("old"),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("requiredField"), "").MarkAlpha(),
	})
}

func TestBeta(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		RequiredField:     ptr.To("val"),
		RequiredFieldBeta: ptr.To("val"),
	}).ExpectValid()

	// Test failures marked as beta
	st.Value(&Struct{
		RequiredField:     ptr.To("val"),
		RequiredFieldBeta: nil,
	}).OldValue(&Struct{
		RequiredField:     ptr.To("old"),
		RequiredFieldBeta: ptr.To("old"),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Required(field.NewPath("requiredFieldBeta"), "").MarkBeta(),
	})
}
