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

package multipleof

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Test with values that are NOT multiples of 5
	st.Value(&Struct{
		IntField:        1,
		IntPtrField:     ptr.To(2),
		Int16Field:      3,
		Int32Field:      4,
		Int64Field:      6,
		UintField:       7,
		Uint16Field:     8,
		Uint32Field:     9,
		Uint64Field:     11,
		UintPtrField:    ptr.To(uint(12)),
		TypedefField:    IntType(1),
		TypedefPtrField: ptr.To(IntType(2)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), nil, ""),
		field.Invalid(field.NewPath("intPtrField"), nil, ""),
		field.Invalid(field.NewPath("int16Field"), nil, ""),
		field.Invalid(field.NewPath("int32Field"), nil, ""),
		field.Invalid(field.NewPath("int64Field"), nil, ""),
		field.Invalid(field.NewPath("uintField"), nil, ""),
		field.Invalid(field.NewPath("uint16Field"), nil, ""),
		field.Invalid(field.NewPath("uint32Field"), nil, ""),
		field.Invalid(field.NewPath("uint64Field"), nil, ""),
		field.Invalid(field.NewPath("uintPtrField"), nil, ""),
		field.Invalid(field.NewPath("typedefField"), nil, ""),
		field.Invalid(field.NewPath("typedefPtrField"), nil, ""),
	})

	// Test validation ratcheting
	st.Value(&Struct{
		IntField:        1,
		IntPtrField:     ptr.To(2),
		Int16Field:      3,
		Int32Field:      4,
		Int64Field:      6,
		UintField:       7,
		Uint16Field:     8,
		Uint32Field:     9,
		Uint64Field:     11,
		UintPtrField:    ptr.To(uint(12)),
		TypedefField:    IntType(1),
		TypedefPtrField: ptr.To(IntType(2)),
	}).OldValue(&Struct{
		IntField:        1,
		IntPtrField:     ptr.To(2),
		Int16Field:      3,
		Int32Field:      4,
		Int64Field:      6,
		UintField:       7,
		Uint16Field:     8,
		Uint32Field:     9,
		Uint64Field:     11,
		UintPtrField:    ptr.To(uint(12)),
		TypedefField:    IntType(1),
		TypedefPtrField: ptr.To(IntType(2)),
	}).ExpectValid()

	// Test with values that ARE multiples of 5 for regular fields, and multiples of 10 for typedefs
	st.Value(&Struct{
		IntField:        5,
		IntPtrField:     ptr.To(10),
		Int16Field:      15,
		Int32Field:      20,
		Int64Field:      25,
		UintField:       30,
		Uint16Field:     35,
		Uint32Field:     40,
		Uint64Field:     45,
		UintPtrField:    ptr.To(uint(50)),
		TypedefField:    IntType(10),
		TypedefPtrField: ptr.To(IntType(20)),
	}).ExpectValid()
}
