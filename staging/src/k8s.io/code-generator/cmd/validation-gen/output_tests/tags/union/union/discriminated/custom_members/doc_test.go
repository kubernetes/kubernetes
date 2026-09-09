/*
Copyright 2024 The Kubernetes Authors.

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

package custommembers

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{D: DM1, M1: &M1{}}).ExpectValid()
	st.Value(&Struct{D: DM2, M2: &M2{}}).ExpectValid()

	st.Value(&Struct{D: DM2, M1: &M1{}, M2: &M2{}}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m1"), nil, "may only be specified when"),
	}.WithOrigin("union"))

	st.Value(&Struct{D: DM1}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m1"), nil, "must be specified when"),
	}.WithOrigin("union"))

	// Test validation ratcheting
	st.Value(&Struct{D: DM2, M1: &M1{}, M2: &M2{}}).OldValue(&Struct{D: DM2, M1: &M1{}, M2: &M2{}}).ExpectValid()
	st.Value(&Struct{D: DM1}).OldValue(&Struct{D: DM1}).ExpectValid()
}
