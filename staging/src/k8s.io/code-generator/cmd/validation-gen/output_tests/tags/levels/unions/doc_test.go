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

package unions

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestAlpha(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		D:  DM1,
		M1: &M1{},
	}).ExpectValid()

	st.Value(&Struct{
		D:  DM1,
		M1: nil, // required by discriminator
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(field.NewPath("m1"), nil, "").WithOrigin("union").MarkAlpha(),
	})

	st.Value(&MyStruct{
		Z1: &Z1{},
		Z2: &Z2{},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(nil, &MyStruct{
			Z1: &Z1{}, Z2: &Z2{},
		}, "only one of z1, z2 may be specified").WithOrigin("zeroOrOneOf").MarkAlpha(),
	})
}

func TestBeta(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&UnionStructBeta{
		DBeta:  BetaDM1,
		M1Beta: &BetaM1{},
	}).ExpectValid()

	st.Value(&UnionStructBeta{
		DBeta:  BetaDM1,
		M1Beta: nil, // required by discriminator
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(field.NewPath("m1Beta"), nil, "").WithOrigin("union").MarkBeta(),
	})

	st.Value(&MyStructBeta{
		Z1Beta: &BetaZ1{},
		Z2Beta: &BetaZ2{},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Invalid(nil, &MyStructBeta{
			Z1Beta: &BetaZ1{}, Z2Beta: &BetaZ2{},
		}, "only one of z1Beta, z2Beta may be specified").WithOrigin("zeroOrOneOf").MarkBeta(),
	})
}
