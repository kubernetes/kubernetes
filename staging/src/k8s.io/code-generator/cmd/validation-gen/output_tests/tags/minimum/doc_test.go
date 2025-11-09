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

package minimum

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// all zero values
		IntPtrField:     ptr.To(0),
		UintPtrField:    ptr.To(uint(0)),
		TypedefPtrField: ptr.To(IntType(0)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), nil, ""),
		field.Invalid(field.NewPath("intPtrField"), nil, ""),
		field.Invalid(field.NewPath("int16Field"), nil, ""),
		field.Invalid(field.NewPath("int32Field"), nil, ""),
		field.Invalid(field.NewPath("int64Field"), nil, ""),
		field.Invalid(field.NewPath("uintField"), nil, ""),
		field.Invalid(field.NewPath("uintPtrField"), nil, ""),
		field.Invalid(field.NewPath("uint16Field"), nil, ""),
		field.Invalid(field.NewPath("uint32Field"), nil, ""),
		field.Invalid(field.NewPath("uint64Field"), nil, ""),
		field.Invalid(field.NewPath("typedefField"), nil, ""),
		field.Invalid(field.NewPath("typedefPtrField"), nil, ""),
	})
	// Test validation ratcheting
	st.Value(&Struct{
		IntPtrField:     ptr.To(0),
		UintPtrField:    ptr.To(uint(0)),
		TypedefPtrField: ptr.To(IntType(0)),
	}).OldValue(&Struct{
		IntPtrField:     ptr.To(0),
		UintPtrField:    ptr.To(uint(0)),
		TypedefPtrField: ptr.To(IntType(0)),
	}).ExpectValid()

	st.Value(&Struct{
		IntField:        1,
		IntPtrField:     ptr.To(1),
		Int16Field:      1,
		Int32Field:      1,
		Int64Field:      1,
		UintField:       1,
		Uint16Field:     1,
		Uint32Field:     1,
		Uint64Field:     1,
		UintPtrField:    ptr.To(uint(1)),
		TypedefField:    IntType(1),
		TypedefPtrField: ptr.To(IntType(1)),
	}).ExpectValid()
}
