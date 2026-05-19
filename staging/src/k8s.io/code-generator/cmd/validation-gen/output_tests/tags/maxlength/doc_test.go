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

package maxlength

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
		Max10Field:                      strings.Repeat("x", 1),
		Max10PtrField:                   ptr.To(strings.Repeat("x", 1)),
		Max10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 1)),
		Max10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Max10ValidatedTypedefField:      Max10Type(strings.Repeat("x", 1)),
		Max10ValidatedTypedefPtrField:   ptr.To(Max10Type(strings.Repeat("x", 1))),
	}).ExpectValid()

	st.Value(&Struct{
		Max10Field:                      strings.Repeat("x", 9),
		Max10PtrField:                   ptr.To(strings.Repeat("x", 9)),
		Max10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 9)),
		Max10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 9))),
		Max10ValidatedTypedefField:      Max10Type(strings.Repeat("x", 9)),
		Max10ValidatedTypedefPtrField:   ptr.To(Max10Type(strings.Repeat("x", 9))),
	}).ExpectValid()

	st.Value(&Struct{
		Max10Field:                      strings.Repeat("x", 10),
		Max10PtrField:                   ptr.To(strings.Repeat("x", 10)),
		Max10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 10)),
		Max10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 10))),
		Max10ValidatedTypedefField:      Max10Type(strings.Repeat("x", 10)),
		Max10ValidatedTypedefPtrField:   ptr.To(Max10Type(strings.Repeat("x", 10))),
	}).ExpectValid()

	testVal := &Struct{
		Max0Field:                       strings.Repeat("x", 1),
		Max0PtrField:                    ptr.To(strings.Repeat("x", 1)),
		Max10Field:                      strings.Repeat("x", 11),
		Max10PtrField:                   ptr.To(strings.Repeat("x", 11)),
		Max0UnvalidatedTypedefField:     UnvalidatedStringType(strings.Repeat("x", 1)),
		Max0UnvalidatedTypedefPtrField:  ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Max10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 11)),
		Max10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 11))),
		Max0ValidatedTypedefField:       Max0Type(strings.Repeat("x", 1)),
		Max0ValidatedTypedefPtrField:    ptr.To(Max0Type(strings.Repeat("x", 1))),
		Max10ValidatedTypedefField:      Max10Type(strings.Repeat("x", 11)),
		Max10ValidatedTypedefPtrField:   ptr.To(Max10Type(strings.Repeat("x", 11))),
	}
	st.Value(testVal).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooLong(field.NewPath("max0Field"), nil, 0),
		field.TooLong(field.NewPath("max0PtrField"), nil, 0),
		field.TooLong(field.NewPath("max10Field"), nil, 10),
		field.TooLong(field.NewPath("max10PtrField"), nil, 10),
		field.TooLong(field.NewPath("max0UnvalidatedTypedefField"), nil, 0),
		field.TooLong(field.NewPath("max0UnvalidatedTypedefPtrField"), nil, 0),
		field.TooLong(field.NewPath("max10UnvalidatedTypedefField"), nil, 10),
		field.TooLong(field.NewPath("max10UnvalidatedTypedefPtrField"), nil, 10),
		field.TooLong(field.NewPath("max0ValidatedTypedefField"), nil, 0),
		field.TooLong(field.NewPath("max0ValidatedTypedefPtrField"), nil, 0),
		field.TooLong(field.NewPath("max10ValidatedTypedefField"), nil, 10),
		field.TooLong(field.NewPath("max10ValidatedTypedefPtrField"), nil, 10),
	})

	// Test validation ratcheting
	st.Value(&Struct{
		Max0Field:                       strings.Repeat("x", 1),
		Max0PtrField:                    ptr.To(strings.Repeat("x", 1)),
		Max10Field:                      strings.Repeat("x", 11),
		Max10PtrField:                   ptr.To(strings.Repeat("x", 11)),
		Max0UnvalidatedTypedefField:     UnvalidatedStringType(strings.Repeat("x", 1)),
		Max0UnvalidatedTypedefPtrField:  ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Max10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 11)),
		Max10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 11))),
		Max0ValidatedTypedefField:       Max0Type(strings.Repeat("x", 1)),
		Max0ValidatedTypedefPtrField:    ptr.To(Max0Type(strings.Repeat("x", 1))),
		Max10ValidatedTypedefField:      Max10Type(strings.Repeat("x", 11)),
		Max10ValidatedTypedefPtrField:   ptr.To(Max10Type(strings.Repeat("x", 11))),
	}).OldValue(&Struct{
		Max0Field:                       strings.Repeat("x", 1),
		Max0PtrField:                    ptr.To(strings.Repeat("x", 1)),
		Max10Field:                      strings.Repeat("x", 11),
		Max10PtrField:                   ptr.To(strings.Repeat("x", 11)),
		Max0UnvalidatedTypedefField:     UnvalidatedStringType(strings.Repeat("x", 1)),
		Max0UnvalidatedTypedefPtrField:  ptr.To(UnvalidatedStringType(strings.Repeat("x", 1))),
		Max10UnvalidatedTypedefField:    UnvalidatedStringType(strings.Repeat("x", 11)),
		Max10UnvalidatedTypedefPtrField: ptr.To(UnvalidatedStringType(strings.Repeat("x", 11))),
		Max0ValidatedTypedefField:       Max0Type(strings.Repeat("x", 1)),
		Max0ValidatedTypedefPtrField:    ptr.To(Max0Type(strings.Repeat("x", 1))),
		Max10ValidatedTypedefField:      Max10Type(strings.Repeat("x", 11)),
		Max10ValidatedTypedefPtrField:   ptr.To(Max10Type(strings.Repeat("x", 11))),
	}).ExpectValid()
}
