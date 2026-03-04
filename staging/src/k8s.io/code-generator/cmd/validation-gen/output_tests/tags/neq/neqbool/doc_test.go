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

package neqbool

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		NeqTrueField:          false,
		NeqFalsePtrField:      ptr.To(true),
		ValidatedTypedefField: false,
	}).ExpectValid()

	st.Value(&Struct{
		NeqTrueField:          false,
		NeqFalsePtrField:      nil,
		ValidatedTypedefField: false,
	}).ExpectValid()

	invalid := &Struct{
		NeqTrueField:          true,
		NeqFalsePtrField:      ptr.To(false),
		ValidatedTypedefField: true,
	}

	st.Value(invalid).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("neqTrueField"), nil, "").WithOrigin("neq"),
		field.Invalid(field.NewPath("neqFalsePtrField"), nil, "").WithOrigin("neq"),
		field.Invalid(field.NewPath("validatedTypedefField"), nil, "").WithOrigin("neq"),
	})

	// Test validation ratcheting.
	st.Value(invalid).OldValue(invalid).ExpectValid()
}
