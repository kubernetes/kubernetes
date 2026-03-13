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

package atomicslice

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestAtomicSlice(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&AtomicSliceStruct{
		Standard:        []int{5},
		Alpha:           []int{5},
		Beta:            []int{5},
		AlphaValidation: []int{5},
		BetaValidation:  []int{5},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		// Case: Standard -> Normal Error
		field.Invalid(field.NewPath("standard").Index(0), 5, "").WithOrigin("minimum"),

		// Case: Alpha eachVal -> Alpha Error
		field.Invalid(field.NewPath("Alpha").Index(0), 5, "").WithOrigin("minimum").MarkAlpha(),

		// Case: Beta eachVal -> Beta Error
		field.Invalid(field.NewPath("Beta").Index(0), 5, "").WithOrigin("minimum").MarkBeta(),

		// Case: Standard eachVal, Alpha validation -> Alpha Error
		field.Invalid(field.NewPath("AlphaValidation").Index(0), 5, "").WithOrigin("minimum").MarkAlpha(),

		// Case: Standard eachVal, Beta validation -> Beta Error
		field.Invalid(field.NewPath("BetaValidation").Index(0), 5, "").WithOrigin("minimum").MarkBeta(),
	})
}
