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

package eqstring

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		StringField:              "expected-string",
		StringPtrField:           new("expected-pointer"),
		StringTypedefField:       "expected-typedef",
		StringTypedefPtrField:    new(StringType("expected-typedef-pointer")),
		ValidatedTypedefField:    "expected-on-type",
		ValidatedTypedefPtrField: new(ValidatedStringType("expected-on-type")),
	}).ExpectValid()

	st.Value(&Struct{
		StringField:              "expected-string",
		StringPtrField:           nil,
		StringTypedefField:       "expected-typedef",
		StringTypedefPtrField:    nil,
		ValidatedTypedefField:    "expected-on-type",
		ValidatedTypedefPtrField: nil,
	}).ExpectValid()

	invalid := &Struct{
		StringField:              "other-string",
		StringPtrField:           new("other-pointer"),
		StringTypedefField:       "other-typedef",
		StringTypedefPtrField:    new(StringType("other-typedef-pointer")),
		ValidatedTypedefField:    "other-on-type",
		ValidatedTypedefPtrField: new(ValidatedStringType("other-on-type")),
	}

	st.Value(invalid).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.NotSupported(field.NewPath("stringField"), nil, []string{"expected-string"}),
		field.NotSupported(field.NewPath("stringPtrField"), nil, []string{"expected-pointer"}),
		field.NotSupported(field.NewPath("stringTypedefField"), nil, []string{"expected-typedef"}),
		field.NotSupported(field.NewPath("stringTypedefPtrField"), nil, []string{"expected-typedef-pointer"}),
		field.NotSupported(field.NewPath("validatedTypedefField"), nil, []string{"expected-on-type"}),
		field.NotSupported(field.NewPath("validatedTypedefPtrField"), nil, []string{"expected-on-type"}),
	})

	// Test validation ratcheting.
	st.Value(invalid).OldValue(invalid).ExpectValid()
}
