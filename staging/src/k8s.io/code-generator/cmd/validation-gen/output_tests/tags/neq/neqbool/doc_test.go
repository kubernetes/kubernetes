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

	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		Enabled:               true,
		DisabledPtr:           ptr.To(false),
		ValidatedTypedefField: true,
	}).ExpectInvalid(
		field.Invalid(field.NewPath("enabled"), true, content.NEQError(true)),
		field.Invalid(field.NewPath("disabledPtr"), false, content.NEQError(false)),
		field.Invalid(field.NewPath("validatedTypedefField"), ValidatedBoolType(true), content.NEQError(ValidatedBoolType(true))),
	)

	// Test validation ratcheting
	st.Value(&Struct{
		Enabled:               true,
		DisabledPtr:           ptr.To(false),
		ValidatedTypedefField: true,
	}).OldValue(&Struct{
		Enabled:               true,
		DisabledPtr:           ptr.To(false),
		ValidatedTypedefField: true,
	}).ExpectValid()

	st.Value(&Struct{
		Enabled:               false,
		DisabledPtr:           ptr.To(true),
		ValidatedTypedefField: false,
	}).ExpectValid()

	st.Value(&Struct{
		Enabled:               false,
		DisabledPtr:           nil,
		ValidatedTypedefField: false,
	}).ExpectValid()
}
