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

package listmapitem

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestListMapItem(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&ListMapItemStruct{
		StandardItem:    []MapItem{{Key: "foo", Value: 5}},
		AlphaItemTag:    []MapItem{{Key: "foo", Value: 5}},
		AlphaValidation: []MapItem{{Key: "foo", Value: 5}},
		DoubleAlpha:     []MapItem{{Key: "foo", Value: 5}},

		BetaItemTag:    []MapItem{{Key: "foo", Value: 5}},
		BetaValidation: []MapItem{{Key: "foo", Value: 5}},
		DoubleBeta:     []MapItem{{Key: "foo", Value: 5}},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		// Case 1: Standard Item -> Normal Error
		field.Invalid(field.NewPath("standardItem").Index(0).Child("value"), 5, "").WithOrigin("minimum"),

		// Case 2: Alpha Item tag -> Alpha Error
		field.Invalid(field.NewPath("alphaItemTag").Index(0).Child("value"), 5, "").WithOrigin("minimum").MarkAlpha(),

		// Case 3: Alpha validation -> Alpha Error
		field.Invalid(field.NewPath("alphaValidation").Index(0).Child("value"), 5, "").WithOrigin("minimum").MarkAlpha(),

		// Case 4: Double Alpha -> Alpha Error
		field.Invalid(field.NewPath("doubleAlpha").Index(0).Child("value"), 5, "").WithOrigin("minimum").MarkAlpha(),

		// Case 5: Beta Item tag -> Beta Error
		field.Invalid(field.NewPath("betaItemTag").Index(0).Child("value"), 5, "").WithOrigin("minimum").MarkBeta(),

		// Case 6: Beta validation -> Beta Error
		field.Invalid(field.NewPath("betaValidation").Index(0).Child("value"), 5, "").WithOrigin("minimum").MarkBeta(),

		// Case 7: Double Beta -> Beta Error
		field.Invalid(field.NewPath("doubleBeta").Index(0).Child("value"), 5, "").WithOrigin("minimum").MarkBeta(),
	})
}
