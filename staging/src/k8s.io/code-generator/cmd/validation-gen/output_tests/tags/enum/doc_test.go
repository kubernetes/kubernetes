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

package enum

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero vals
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("enum0Field"), Enum0(""), []Enum0{}),
		field.NotSupported(field.NewPath("enum1Field"), Enum1(""), []Enum1{E1V1}),
		field.NotSupported(field.NewPath("enum2Field"), Enum2(""), []Enum2{E2V1, E2V2}),
	})

	st.Value(&Struct{
		Enum0Field:      "",                // no valid value exists
		Enum0PtrField:   ptr.To(Enum0("")), // no valid value exists
		Enum1Field:      E1V1,
		Enum1PtrField:   ptr.To(E1V1),
		Enum2Field:      E2V1,
		Enum2PtrField:   ptr.To(E2V1),
		NotEnumField:    "x",
		NotEnumPtrField: ptr.To(NotEnum("x")),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("enum0Field"), Enum0(""), []Enum0{}),
		field.NotSupported(field.NewPath("enum0PtrField"), Enum0(""), []Enum0{}),
	})

	st.Value(&Struct{
		Enum0Field:      "x",                // no valid value exists
		Enum0PtrField:   ptr.To(Enum0("x")), // no valid value exists
		Enum1Field:      "x",
		Enum1PtrField:   ptr.To(Enum1("x")),
		Enum2Field:      "x",
		Enum2PtrField:   ptr.To(Enum2("x")),
		NotEnumField:    "x",
		NotEnumPtrField: ptr.To(NotEnum("x")),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("enum0Field"), Enum0("x"), []Enum0{}),
		field.NotSupported(field.NewPath("enum0PtrField"), Enum0("x"), []Enum0{}),
		field.NotSupported(field.NewPath("enum1Field"), Enum1("x"), []Enum1{E1V1}),
		field.NotSupported(field.NewPath("enum1PtrField"), Enum1("x"), []Enum1{E1V1}),
		field.NotSupported(field.NewPath("enum2Field"), Enum2("x"), []Enum2{E2V1, E2V2}),
		field.NotSupported(field.NewPath("enum2PtrField"), Enum2("x"), []Enum2{E2V1, E2V2}),
	})
}
