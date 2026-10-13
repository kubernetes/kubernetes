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

package multiple

import (
	"testing"

	field "k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(nil, nil, "must specify one of: `u1m1`, `u1m2`").WithOrigin("union"),
		field.Invalid(nil, nil, "must specify one of: `u2m1`, `u2m2`").WithOrigin("union"),
	})

	st.Value(&Struct{U1M1: &M1{}, U2M1: &M1{}}).ExpectValid()
	st.Value(&Struct{U1M2: &M2{}, U2M2: &M2{}}).ExpectValid()

	st.Value(&Struct{U1M1: &M1{}, U1M2: &M2{}}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(nil, nil, "must specify exactly one of: `u1m1`, `u1m2`").WithOrigin("union"),
		field.Invalid(nil, nil, "must specify one of: `u2m1`, `u2m2`").WithOrigin("union"),
	})

	st.Value(&Struct{U2M1: &M1{}, U2M2: &M2{}}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(nil, nil, "must specify one of: `u1m1`, `u1m2`").WithOrigin("union"),
		field.Invalid(nil, nil, "must specify exactly one of: `u2m1`, `u2m2`").WithOrigin("union"),
	})

	st.Value(&Struct{
		U1M1: &M1{}, U1M2: &M2{},
		U2M1: &M1{}, U2M2: &M2{},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(nil, nil, "must specify exactly one of: `u1m1`, `u1m2`").WithOrigin("union"),
		field.Invalid(nil, nil, "must specify exactly one of: `u2m1`, `u2m2`").WithOrigin("union"),
	})

	// Test validation ratcheting
	st.Value(&Struct{}).OldValue(&Struct{}).ExpectValid()
	st.Value(&Struct{U1M1: &M1{}, U1M2: &M2{}}).OldValue(&Struct{U1M1: &M1{}, U1M2: &M2{}}).ExpectValid()
	st.Value(&Struct{U2M1: &M1{}, U2M2: &M2{}}).OldValue(&Struct{U2M1: &M1{}, U2M2: &M2{}}).ExpectValid()
	st.Value(&Struct{
		U1M1: &M1{}, U1M2: &M2{},
		U2M1: &M1{}, U2M2: &M2{},
	}).OldValue(&Struct{
		U1M1: &M1{}, U1M2: &M2{},
		U2M1: &M1{}, U2M2: &M2{},
	}).ExpectValid()
}
