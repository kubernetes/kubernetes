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

package listset

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestListSet(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&ListSetStruct{
		Set: []ComplexSetItem{
			{StringVal: "abc", Value: 10},
			{StringVal: "abc", Value: 10}, // Duplicate value (alpha)
		},
		BetaSet: []ComplexSetItem{
			{StringVal: "abc", Value: 10},
			{StringVal: "abc", Value: 10}, // Duplicate value (beta)
		},
		ChainedSubfieldSet: []SimpleSetItem{
			{StringVal: "def", Value: 9}, // Value 9 < 10 (alpha)
		},
		SetBetaItem: []ComplexSetItemBeta{
			{StringVal: "ghi", Value: 10},
			{StringVal: "ghi", Value: 10}, // Duplicate value (alpha)
		},
		BetaSetBetaItem: []ComplexSetItemBeta{
			{StringVal: "jkl", Value: 10},
			{StringVal: "jkl", Value: 10}, // Duplicate value (beta)
		},
		ChainedSubfieldSetBeta: []SimpleSetItem{
			{StringVal: "mno", Value: 9}, // Value 9 < 10 (beta)
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		// Set: Duplicate value is SHADOWED
		field.Duplicate(field.NewPath("set").Index(1), ComplexSetItem{StringVal: "abc", Value: 10}).MarkAlpha(),

		// BetaSet: Duplicate value is SHADOWED
		field.Duplicate(field.NewPath("betaSet").Index(1), ComplexSetItem{StringVal: "abc", Value: 10}).MarkBeta(),

		// ChainedSubfieldSet: Value 9 < 10 is SHADOWED
		field.Invalid(field.NewPath("chainedSubfieldSet").Index(0).Child("value"), 9, "").WithOrigin("minimum").MarkAlpha(),

		// SetBetaItem: Duplicate value is SHADOWED (Alpha list)
		field.Duplicate(field.NewPath("setBetaItem").Index(1), ComplexSetItemBeta{StringVal: "ghi", Value: 10}).MarkAlpha(),

		// BetaSetBetaItem: Duplicate value is SHADOWED (Beta list)
		field.Duplicate(field.NewPath("betaSetBetaItem").Index(1), ComplexSetItemBeta{StringVal: "jkl", Value: 10}).MarkBeta(),

		// ChainedSubfieldSetBeta: Value 9 < 10 is SHADOWED (Beta item)
		field.Invalid(field.NewPath("chainedSubfieldSetBeta").Index(0).Child("value"), 9, "").WithOrigin("minimum").MarkBeta(),
	})
}
