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

// +k8s:validation-gen=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package.
package subfield

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestStruct(t *testing.T) {
	st := localSchemeBuilder.Test(t)
	st.Value(&Struct{
		SubStructField: SubStruct{
			IntField:    1,
			IntPtrField: ptr.To(1),
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("subStructField").Child("intField"), 1, "field IntField"),
		field.Invalid(field.NewPath("subStructField").Child("intPtrField"), 1, "field IntPtrField"),
	})

	st.Value(&StructWithSubfield{
		IntField:    1,
		IntPtrField: ptr.To(1),
	}).OldValue(&StructWithSubfield{
		IntField:    1,
		IntPtrField: ptr.To(1),
	}).ExpectValid()
}

func TestStructWithSubfield(t *testing.T) {
	st := localSchemeBuilder.Test(t)
	st.Value(&StructWithSubfield{
		IntField:    1,
		IntPtrField: ptr.To(1),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), 1, "field IntField"),
		field.Invalid(field.NewPath("intPtrField"), 1, "field IntPtrField"),
	})

	st.Value(&StructWithSubfield{
		TypeMeta:    1,
		IntField:    1,
		IntPtrField: ptr.To(1),
	}).OldValue(&StructWithSubfield{
		TypeMeta:    1,
		IntField:    1,
		IntPtrField: ptr.To(1),
	}).ExpectValid()
}
