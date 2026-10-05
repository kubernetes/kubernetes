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

package minlength

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values
	}).ExpectValid()

	st.Value(&Struct{
		Min2Field:                          strings.Repeat("x", 1),
		Min2PtrField:                       ptr.To(strings.Repeat("x", 1)),
		Min2UnvalidatedTypedefField:        UnvalidatedStringType(strings.Repeat("x", 1)),
		Min2UnvalidatedTypedefPtrField:     ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Min2ValidatedTypedefField:          Min2Type(strings.Repeat("x", 1)),
		Min2ValidatedTypedefPtrField:       ptr.To(Min2Type(strings.Repeat("x", 1))),
		Min10Field:                         strings.Repeat("x", 1),
		Min10PtrField:                      ptr.To(strings.Repeat("x", 1)),
		Min10UnvalidatedTypedefField:       UnvalidatedStringType(strings.Repeat("x", 1)),
		Min10UnvalidatedTypedefPtrField:    ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Min10ValidatedTypedefField:         Min10Type(strings.Repeat("x", 1)),
		Min10ValidatedTypedefPtrField:      ptr.To(Min10Type(strings.Repeat("x", 1))),
		Min2UnvalidatedStringAliasField:    strings.Repeat("x", 1),
		Min2UnvalidatedStringAliasPtrField: ptr.To(strings.Repeat("x", 1)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooShort(field.NewPath("min10Field"), "", 10),
		field.TooShort(field.NewPath("min10PtrField"), "", 10),
		field.TooShort(field.NewPath("min10UnvalidatedTypedefField"), "", 10),
		field.TooShort(field.NewPath("min10UnvalidatedTypedefPtrField"), "", 10),
		field.TooShort(field.NewPath("min10ValidatedTypedefField"), "", 10),
		field.TooShort(field.NewPath("min10ValidatedTypedefPtrField"), "", 10),
		field.TooShort(field.NewPath("min2Field"), "x", 2),
		field.TooShort(field.NewPath("min2PtrField"), "x", 2),
		field.TooShort(field.NewPath("min2UnvalidatedTypedefField"), "x", 2),
		field.TooShort(field.NewPath("min2UnvalidatedTypedefPtrField"), "x", 2),
		field.TooShort(field.NewPath("min2ValidatedTypedefField"), "x", 2),
		field.TooShort(field.NewPath("min2ValidatedTypedefPtrField"), "x", 2),
		field.TooShort(field.NewPath("min2UnvalidatedStringAliasField"), "x", 2),
		field.TooShort(field.NewPath("min2UnvalidatedStringAliasPtrField"), "x", 2),
	})

	st.Value(&Struct{
		Min2Field:                          strings.Repeat("x", 1),
		Min2PtrField:                       ptr.To(strings.Repeat("x", 1)),
		Min2UnvalidatedTypedefField:        UnvalidatedStringType(strings.Repeat("x", 1)),
		Min2UnvalidatedTypedefPtrField:     ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Min2ValidatedTypedefField:          Min2Type(strings.Repeat("x", 1)),
		Min2ValidatedTypedefPtrField:       ptr.To(Min2Type(strings.Repeat("x", 1))),
		Min10Field:                         strings.Repeat("x", 9),
		Min10PtrField:                      ptr.To(strings.Repeat("x", 9)),
		Min10UnvalidatedTypedefField:       UnvalidatedStringType(strings.Repeat("x", 9)),
		Min10UnvalidatedTypedefPtrField:    ptr.To(UnvalidatedStringType(strings.Repeat("x", 9))),
		Min10ValidatedTypedefField:         Min10Type(strings.Repeat("x", 9)),
		Min10ValidatedTypedefPtrField:      ptr.To(Min10Type(strings.Repeat("x", 9))),
		Min2UnvalidatedStringAliasField:    strings.Repeat("x", 1),
		Min2UnvalidatedStringAliasPtrField: ptr.To(strings.Repeat("x", 1)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooShort(field.NewPath("min10Field"), "", 10),
		field.TooShort(field.NewPath("min10PtrField"), "", 10),
		field.TooShort(field.NewPath("min10UnvalidatedTypedefField"), "", 10),
		field.TooShort(field.NewPath("min10UnvalidatedTypedefPtrField"), "", 10),
		field.TooShort(field.NewPath("min10ValidatedTypedefField"), "", 10),
		field.TooShort(field.NewPath("min10ValidatedTypedefPtrField"), "", 10),
		field.TooShort(field.NewPath("min2Field"), "x", 2),
		field.TooShort(field.NewPath("min2PtrField"), "x", 2),
		field.TooShort(field.NewPath("min2UnvalidatedTypedefField"), "x", 2),
		field.TooShort(field.NewPath("min2UnvalidatedTypedefPtrField"), "x", 2),
		field.TooShort(field.NewPath("min2ValidatedTypedefField"), "x", 2),
		field.TooShort(field.NewPath("min2ValidatedTypedefPtrField"), "x", 2),
		field.TooShort(field.NewPath("min2UnvalidatedStringAliasField"), "x", 2),
		field.TooShort(field.NewPath("min2UnvalidatedStringAliasPtrField"), "x", 2),
	})

	st.Value(&Struct{
		Min2Field:                          strings.Repeat("x", 2),
		Min2PtrField:                       ptr.To(strings.Repeat("x", 2)),
		Min2UnvalidatedTypedefField:        UnvalidatedStringType(strings.Repeat("x", 2)),
		Min2UnvalidatedTypedefPtrField:     ptr.To(UnvalidatedStringType(strings.Repeat("x", 2))),
		Min2ValidatedTypedefField:          Min2Type(strings.Repeat("x", 2)),
		Min10Field:                         strings.Repeat("x", 10),
		Min10PtrField:                      ptr.To(strings.Repeat("x", 10)),
		Min10UnvalidatedTypedefField:       UnvalidatedStringType(strings.Repeat("x", 10)),
		Min10UnvalidatedTypedefPtrField:    ptr.To(UnvalidatedStringType(strings.Repeat("x", 10))),
		Min10ValidatedTypedefField:         Min10Type(strings.Repeat("x", 10)),
		Min10ValidatedTypedefPtrField:      ptr.To(Min10Type(strings.Repeat("x", 10))),
		Min2UnvalidatedStringAliasField:    strings.Repeat("x", 2),
		Min2UnvalidatedStringAliasPtrField: ptr.To(strings.Repeat("x", 2)),
	}).ExpectValid()

	testVal := &Struct{
		Min2Field:                          strings.Repeat("x", 3),
		Min2PtrField:                       ptr.To(strings.Repeat("x", 3)),
		Min10Field:                         strings.Repeat("x", 11),
		Min10PtrField:                      ptr.To(strings.Repeat("x", 11)),
		Min2UnvalidatedTypedefField:        UnvalidatedStringType(strings.Repeat("x", 3)),
		Min2UnvalidatedTypedefPtrField:     ptr.To(UnvalidatedStringType(strings.Repeat("x", 3))),
		Min10UnvalidatedTypedefField:       UnvalidatedStringType(strings.Repeat("x", 11)),
		Min10UnvalidatedTypedefPtrField:    ptr.To(UnvalidatedStringType(strings.Repeat("x", 11))),
		Min2ValidatedTypedefField:          Min2Type(strings.Repeat("x", 3)),
		Min2ValidatedTypedefPtrField:       ptr.To(Min2Type(strings.Repeat("x", 3))),
		Min10ValidatedTypedefField:         Min10Type(strings.Repeat("x", 11)),
		Min10ValidatedTypedefPtrField:      ptr.To(Min10Type(strings.Repeat("x", 11))),
		Min2UnvalidatedStringAliasField:    strings.Repeat("x", 3),
		Min2UnvalidatedStringAliasPtrField: ptr.To(strings.Repeat("x", 3)),
	}
	st.Value(testVal).ExpectValid()

	// Test validation ratcheting
	st.Value(&Struct{
		Min2Field:                       strings.Repeat("x", 2),
		Min2PtrField:                    ptr.To(strings.Repeat("x", 2)),
		Min10Field:                      strings.Repeat("x", 11),
		Min10PtrField:                   ptr.To(strings.Repeat("x", 11)),
		Min2UnvalidatedTypedefField:     UnvalidatedStringType(strings.Repeat("x", 2)),
		Min2UnvalidatedTypedefPtrField:  ptr.To(UnvalidatedStringType(strings.Repeat("x", 2))),
		Min10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 11)),
		Min10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 11))),
		Min2ValidatedTypedefField:       Min2Type(strings.Repeat("x", 2)),
		Min2ValidatedTypedefPtrField:    ptr.To(Min2Type(strings.Repeat("x", 2))),
		Min10ValidatedTypedefField:      Min10Type(strings.Repeat("x", 11)),
		Min10ValidatedTypedefPtrField:   ptr.To(Min10Type(strings.Repeat("x", 11))),
	}).OldValue(&Struct{
		Min2Field:                       strings.Repeat("x", 1),
		Min2PtrField:                    ptr.To(strings.Repeat("x", 1)),
		Min10Field:                      strings.Repeat("x", 9),
		Min10PtrField:                   ptr.To(strings.Repeat("x", 1)),
		Min2UnvalidatedTypedefField:     UnvalidatedStringType(strings.Repeat("x", 1)),
		Min2UnvalidatedTypedefPtrField:  ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Min10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 9)),
		Min10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 9))),
		Min2ValidatedTypedefField:       Min2Type(strings.Repeat("x", 1)),
		Min2ValidatedTypedefPtrField:    ptr.To(Min2Type(strings.Repeat("x", 1))),
		Min10ValidatedTypedefField:      Min10Type(strings.Repeat("x", 9)),
		Min10ValidatedTypedefPtrField:   ptr.To(Min10Type(strings.Repeat("x", 9))),
	}).ExpectValid()
}
