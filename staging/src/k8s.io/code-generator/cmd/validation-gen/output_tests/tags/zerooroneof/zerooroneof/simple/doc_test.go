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

package simple

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Empty union is valid for zeroOrOneOf
	st.Value(&Struct{}).ExpectValid()

	st.Value(&Struct{M1: &M1{}}).ExpectValid()
	st.Value(&Struct{M2: &M2{}}).ExpectValid()
	st.Value(&Struct{M3: "a string"}).ExpectValid()
	st.Value(&Struct{M4: ptr.To("a string")}).ExpectValid()

	st.Value(&Struct{M1: &M1{}, M2: &M2{}}).ExpectInvalid(
		field.Invalid(nil, "{m1, m2}", "must specify at most one of: `m1`, `m2`, `m3`, `m4`"),
	)
	st.Value(&Struct{M1: &M1{}, M3: "a string"}).ExpectInvalid(
		field.Invalid(nil, "{m1, m3}", "must specify at most one of: `m1`, `m2`, `m3`, `m4`"),
	)
	st.Value(&Struct{M1: &M1{}, M4: ptr.To("a string")}).ExpectInvalid(
		field.Invalid(nil, "{m1, m4}", "must specify at most one of: `m1`, `m2`, `m3`, `m4`"),
	)

	// Update only considers whether a field was set, not the value.
	st.Value(&Struct{M3: "a string"}).OldValue(&Struct{M3: "different string"}).ExpectValid()

	// Test validation ratcheting
	st.Value(&Struct{}).OldValue(&Struct{}).ExpectValid()
	st.Value(&Struct{M1: &M1{}, M2: &M2{}}).OldValue(&Struct{M1: &M1{}, M2: &M2{}}).ExpectValid()
	st.Value(&Struct{M3: "a string", M2: &M2{}}).OldValue(&Struct{M3: "different string", M2: &M2{}}).ExpectValid()
}
