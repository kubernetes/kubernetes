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

package builtins

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		IntPtrField:  new(1),
		StringField:  "abc",
		EnumField:    EnumA,
		ListMapField: []Item{{Name: "a"}, {Name: "b"}},
	}).ExpectValid()

	st.Value(&Struct{
		StringField:  "abcd",
		EnumField:    "C",
		ListMapField: []Item{{Name: "abcd"}, {Name: "abcd"}},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Required(field.NewPath("intPtrField"), ""),
		field.TooLong(field.NewPath("stringField"), nil, 3).WithOrigin("maxLength"),
		field.NotSupported(field.NewPath("enumField"), Enum(""), []Enum{}),
		field.Duplicate(field.NewPath("listMapField").Index(1), nil),
		field.TooLong(field.NewPath("listMapField").Index(0).Child("name"), nil, 3).WithOrigin("maxLength"),
		field.TooLong(field.NewPath("listMapField").Index(1).Child("name"), nil, 3).WithOrigin("maxLength"),
	})

	st.Value(&Struct{
		IntPtrField: new(0),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("intPtrField"), nil, "").WithOrigin("minimum"),
	})

	st.Value(&Struct{
		IntPtrField:    new(1),
		ImmutableField: "new",
	}).OldValue(&Struct{
		IntPtrField:    new(1),
		ImmutableField: "old",
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("immutableField"), nil, "").WithOrigin("immutable"),
	})
}
