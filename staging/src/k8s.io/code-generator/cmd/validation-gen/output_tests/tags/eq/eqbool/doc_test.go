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

package eqbool

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		EqTrueField:           true,
		EqFalsePtrField:       new(false),
		ValidatedTypedefField: true,
	}).ExpectValid()

	st.Value(&Struct{
		EqTrueField:           true,
		EqFalsePtrField:       nil,
		ValidatedTypedefField: true,
	}).ExpectValid()

	invalid := &Struct{
		EqTrueField:           false,
		EqFalsePtrField:       new(true),
		ValidatedTypedefField: false,
	}

	st.Value(invalid).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.NotSupported(field.NewPath("eqTrueField"), nil, []string{"true"}),
		field.NotSupported(field.NewPath("eqFalsePtrField"), nil, []string{"false"}),
		field.NotSupported(field.NewPath("validatedTypedefField"), nil, []string{"true"}),
	})

	// Test validation ratcheting.
	st.Value(invalid).OldValue(invalid).ExpectValid()
}
