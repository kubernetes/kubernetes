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

package unions

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestStructValidation(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid case: D is M1, M1 is set
	st.Value(&Struct{
		Subfield: SubStruct{
			D:  "M1",
			M1: ptr.To(1),
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{})

	// Valid case: D is M2, M2 is set
	st.Value(&Struct{
		Subfield: SubStruct{
			D:  "M2",
			M2: ptr.To(1),
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{})

	// Invalid case: D is M1, but M2 is set
	st.Value(&Struct{
		Subfield: SubStruct{
			D:  "M1",
			M2: ptr.To(1),
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("subfield", "m1"), "", "must be specified when `d` is \"M1\"").WithOrigin("union"),
		field.Invalid(field.NewPath("subfield", "m2"), "", "may only be specified when `d` is \"M2\"").WithOrigin("union"),
	})

	// Invalid case: D is M1, and BOTH are set
	st.Value(&Struct{
		Subfield: SubStruct{
			D:  "M1",
			M1: ptr.To(1),
			M2: ptr.To(1),
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("subfield", "m2"), "", "may only be specified when `d` is \"M2\"").WithOrigin("union"),
	})
}
