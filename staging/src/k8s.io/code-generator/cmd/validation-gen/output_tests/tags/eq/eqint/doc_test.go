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

package eqint

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		IntField:              0,
		IntPtrField:           new(-1),
		IntTypedefField:       42,
		ValidatedTypedefField: 100,
	}).ExpectValid()

	st.Value(&Struct{
		IntField:              0,
		IntPtrField:           nil,
		IntTypedefField:       42,
		ValidatedTypedefField: 100,
	}).ExpectValid()

	invalid := &Struct{
		IntField:              1,
		IntPtrField:           new(0),
		IntTypedefField:       41,
		ValidatedTypedefField: 99,
	}

	st.Value(invalid).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.NotSupported(field.NewPath("intField"), nil, []string{"0"}),
		field.NotSupported(field.NewPath("intPtrField"), nil, []string{"-1"}),
		field.NotSupported(field.NewPath("intTypedefField"), nil, []string{"42"}),
		field.NotSupported(field.NewPath("validatedTypedefField"), nil, []string{"100"}),
	})

	// Test validation ratcheting.
	st.Value(invalid).OldValue(invalid).ExpectValid()
}
