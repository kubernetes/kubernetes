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

package subfields

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestAlpha(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		Subfield:     SubStruct{Inner: 5},
		SubfieldBeta: SubStruct{Inner: 5},
	}).ExpectValid()

	// Test failures marked as alpha
	st.Value(&Struct{
		Subfield:     SubStruct{Inner: 1},
		SubfieldBeta: SubStruct{Inner: 5},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(field.NewPath("subfield", "inner"), 1, "").WithOrigin("minimum").MarkAlpha(),
	})
}

func TestBeta(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		Subfield:     SubStruct{Inner: 5},
		SubfieldBeta: SubStruct{Inner: 5},
	}).ExpectValid()

	// Test failures marked as beta
	st.Value(&Struct{
		Subfield:     SubStruct{Inner: 5},
		SubfieldBeta: SubStruct{Inner: 1},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(field.NewPath("subfieldBeta", "inner"), 1, "").WithOrigin("minimum").MarkBeta(),
	})
}
