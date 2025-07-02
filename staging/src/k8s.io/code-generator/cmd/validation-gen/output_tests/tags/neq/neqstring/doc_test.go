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

package neqstring

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		StringField:              "allowed-string",
		StringPtrField:           ptr.To("allowed-pointer"),
		StringTypedefField:       "allowed-typedef",
		StringTypedefPtrField:    ptr.To(StringType("allowed-typedef-pointer")),
		ValidatedTypedefField:    "allowed-on-type",
		ValidatedTypedefPtrField: ptr.To(ValidatedStringType("allowed-on-type-ptr")),
	}).ExpectValid()

	st.Value(&Struct{
		StringField:              "allowed-string",
		StringPtrField:           nil,
		StringTypedefField:       "allowed-typedef",
		StringTypedefPtrField:    nil,
		ValidatedTypedefField:    "allowed-on-type",
		ValidatedTypedefPtrField: nil,
	}).ExpectValid()

	invalid := &Struct{
		StringField:              "disallowed-string",
		StringPtrField:           ptr.To("disallowed-pointer"),
		StringTypedefField:       "disallowed-typedef",
		StringTypedefPtrField:    ptr.To(StringType("disallowed-typedef-pointer")),
		ValidatedTypedefField:    "disallowed-on-type",
		ValidatedTypedefPtrField: ptr.To(ValidatedStringType("disallowed-on-type")),
	}

	st.Value(invalid).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("stringField"), "disallowed-string", content.NEQError("disallowed-string")).WithOrigin("neq"),
		field.Invalid(field.NewPath("stringPtrField"), "disallowed-pointer", content.NEQError("disallowed-pointer")).WithOrigin("neq"),
		field.Invalid(field.NewPath("stringTypedefField"), StringType("disallowed-typedef"), content.NEQError(StringType("disallowed-typedef"))).WithOrigin("neq"),
		field.Invalid(field.NewPath("stringTypedefPtrField"), StringType("disallowed-typedef-pointer"), content.NEQError(StringType("disallowed-typedef-pointer"))).WithOrigin("neq"),
		field.Invalid(field.NewPath("validatedTypedefField"), ValidatedStringType("disallowed-on-type"), content.NEQError(ValidatedStringType("disallowed-on-type"))).WithOrigin("neq"),
		field.Invalid(field.NewPath("validatedTypedefPtrField"), ValidatedStringType("disallowed-on-type"), content.NEQError(ValidatedStringType("disallowed-on-type"))).WithOrigin("neq"),
	})

	// Test validation ratcheting.
	st.Value(invalid).OldValue(invalid).ExpectValid()
}
