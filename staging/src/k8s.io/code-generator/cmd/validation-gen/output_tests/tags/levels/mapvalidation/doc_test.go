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

package mapvalidation

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestMapValidation(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&MapValidationStruct{
		StandardEachVal: map[string]string{"a": "foo"},
		AlphaEachVal:    map[string]string{"a": "foo"},
		BetaEachVal:     map[string]string{"a": "foo"},
		StandardEachKey: map[string]string{"foo": "a"},
		AlphaEachKey:    map[string]string{"foo": "a"},
		BetaEachKey:     map[string]string{"foo": "a"},
		AlphaValidation: map[string]string{"a": "foo"},
		BetaValidation:  map[string]string{"a": "foo"},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		// Case: Standard eachVal -> Normal Error
		field.TooLong(field.NewPath("standardEachVal").Key("a"), "foo", 2).WithOrigin("maxLength"),

		// Case: Alpha eachVal -> Alpha Error
		field.TooLong(field.NewPath("AlphaEachVal").Key("a"), "foo", 2).WithOrigin("maxLength").MarkAlpha(),

		// Case: Beta eachVal -> Beta Error
		field.TooLong(field.NewPath("BetaEachVal").Key("a"), "foo", 2).WithOrigin("maxLength").MarkBeta(),

		// Case: Standard eachKey -> Normal Error
		field.TooLong(field.NewPath("standardEachKey"), "foo", 2).WithOrigin("maxLength"),

		// Case: Alpha eachKey -> Alpha Error
		field.TooLong(field.NewPath("AlphaEachKey"), "foo", 2).WithOrigin("maxLength").MarkAlpha(),

		// Case: Beta eachKey -> Beta Error
		field.TooLong(field.NewPath("BetaEachKey"), "foo", 2).WithOrigin("maxLength").MarkBeta(),

		// Case: Alpha Validation -> Alpha Error
		field.TooLong(field.NewPath("AlphaValidation").Key("a"), "foo", 2).WithOrigin("maxLength").MarkAlpha(),

		// Case: Beta Validation -> Beta Error
		field.TooLong(field.NewPath("BetaValidation").Key("a"), "foo", 2).WithOrigin("maxLength").MarkBeta(),
	})
}
