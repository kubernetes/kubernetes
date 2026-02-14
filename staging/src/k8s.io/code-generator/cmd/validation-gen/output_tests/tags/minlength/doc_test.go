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
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooShort(field.NewPath("min1Field"), nil, 1),
		field.TooShort(field.NewPath("min10Field"), nil, 10),
		field.TooShort(field.NewPath("min1UnvalidatedTypedefField"), nil, 1),
		field.TooShort(field.NewPath("min10UnvalidatedTypedefField"), nil, 10),
		field.TooShort(field.NewPath("min1ValidatedTypedefField"), nil, 1),
		field.TooShort(field.NewPath("min10ValidatedTypedefField"), nil, 10),
	})

	st.Value(&Struct{
		Min1Field:                       strings.Repeat("x", 1),
		Min1PtrField:                    ptr.To(strings.Repeat("x", 1)),
		Min1UnvalidatedTypedefField:     UnvalidatedStringType(strings.Repeat("x", 1)),
		Min1UnvalidatedTypedefPtrField:  ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Min1ValidatedTypedefField:       Min1Type(strings.Repeat("x", 1)),
		Min1ValidatedTypedefPtrField:    ptr.To(Min1Type(strings.Repeat("x", 1))),
		Min10Field:                      strings.Repeat("x", 1),
		Min10PtrField:                   ptr.To(strings.Repeat("x", 1)),
		Min10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 1)),
		Min10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Min10ValidatedTypedefField:      Min10Type(strings.Repeat("x", 1)),
		Min10ValidatedTypedefPtrField:   ptr.To(Min10Type(strings.Repeat("x", 1))),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooShort(field.NewPath("min10Field"), nil, 10),
		field.TooShort(field.NewPath("min10PtrField"), nil, 10),
		field.TooShort(field.NewPath("min10UnvalidatedTypedefField"), nil, 10),
		field.TooShort(field.NewPath("min10UnvalidatedTypedefPtrField"), nil, 10),
		field.TooShort(field.NewPath("min10ValidatedTypedefField"), nil, 10),
		field.TooShort(field.NewPath("min10ValidatedTypedefPtrField"), nil, 10),
	})

	st.Value(&Struct{
		Min1Field:                       strings.Repeat("x", 1),
		Min1PtrField:                    ptr.To(strings.Repeat("x", 1)),
		Min1UnvalidatedTypedefField:     UnvalidatedStringType(strings.Repeat("x", 1)),
		Min1UnvalidatedTypedefPtrField:  ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Min1ValidatedTypedefField:       Min1Type(strings.Repeat("x", 1)),
		Min1ValidatedTypedefPtrField:    ptr.To(Min1Type(strings.Repeat("x", 1))),
		Min10Field:                      strings.Repeat("x", 9),
		Min10PtrField:                   ptr.To(strings.Repeat("x", 9)),
		Min10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 9)),
		Min10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 9))),
		Min10ValidatedTypedefField:      Min10Type(strings.Repeat("x", 9)),
		Min10ValidatedTypedefPtrField:   ptr.To(Min10Type(strings.Repeat("x", 9))),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooShort(field.NewPath("min10Field"), nil, 10),
		field.TooShort(field.NewPath("min10PtrField"), nil, 10),
		field.TooShort(field.NewPath("min10UnvalidatedTypedefField"), nil, 10),
		field.TooShort(field.NewPath("min10UnvalidatedTypedefPtrField"), nil, 10),
		field.TooShort(field.NewPath("min10ValidatedTypedefField"), nil, 10),
		field.TooShort(field.NewPath("min10ValidatedTypedefPtrField"), nil, 10),
	})

	st.Value(&Struct{
		Min1Field:                       strings.Repeat("x", 1),
		Min1PtrField:                    ptr.To(strings.Repeat("x", 1)),
		Min1UnvalidatedTypedefField:     UnvalidatedStringType(strings.Repeat("x", 1)),
		Min1UnvalidatedTypedefPtrField:  ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Min1ValidatedTypedefField:       Min1Type(strings.Repeat("x", 1)),
		Min10Field:                      strings.Repeat("x", 10),
		Min10PtrField:                   ptr.To(strings.Repeat("x", 10)),
		Min10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 10)),
		Min10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 10))),
		Min10ValidatedTypedefField:      Min10Type(strings.Repeat("x", 10)),
		Min10ValidatedTypedefPtrField:   ptr.To(Min10Type(strings.Repeat("x", 10))),
	}).ExpectValid()

	testVal := &Struct{
		Min1Field:                       strings.Repeat("x", 1),
		Min1PtrField:                    ptr.To(strings.Repeat("x", 1)),
		Min10Field:                      strings.Repeat("x", 11),
		Min10PtrField:                   ptr.To(strings.Repeat("x", 11)),
		Min1UnvalidatedTypedefField:     UnvalidatedStringType(strings.Repeat("x", 1)),
		Min1UnvalidatedTypedefPtrField:  ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Min10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 11)),
		Min10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 11))),
		Min1ValidatedTypedefField:       Min1Type(strings.Repeat("x", 1)),
		Min1ValidatedTypedefPtrField:    ptr.To(Min1Type(strings.Repeat("x", 1))),
		Min10ValidatedTypedefField:      Min10Type(strings.Repeat("x", 11)),
		Min10ValidatedTypedefPtrField:   ptr.To(Min10Type(strings.Repeat("x", 11))),
	}
	st.Value(testVal).ExpectValid()

	// Test validation ratcheting
	st.Value(&Struct{
		Min1Field:                       strings.Repeat("x", 1),
		Min1PtrField:                    ptr.To(strings.Repeat("x", 1)),
		Min10Field:                      strings.Repeat("x", 11),
		Min10PtrField:                   ptr.To(strings.Repeat("x", 11)),
		Min1UnvalidatedTypedefField:     UnvalidatedStringType(strings.Repeat("x", 1)),
		Min1UnvalidatedTypedefPtrField:  ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Min10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 11)),
		Min10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 11))),
		Min1ValidatedTypedefField:       Min1Type(strings.Repeat("x", 1)),
		Min1ValidatedTypedefPtrField:    ptr.To(Min1Type(strings.Repeat("x", 1))),
		Min10ValidatedTypedefField:      Min10Type(strings.Repeat("x", 11)),
		Min10ValidatedTypedefPtrField:   ptr.To(Min10Type(strings.Repeat("x", 11))),
	}).OldValue(&Struct{}).ExpectValid()
}
