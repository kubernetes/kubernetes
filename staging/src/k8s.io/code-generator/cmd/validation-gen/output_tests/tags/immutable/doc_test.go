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

package immutable

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	structA := Struct{
		StringField:                 "aaa",
		StringPtrField:              new("aaa"),
		StructField:                 ComparableStruct{"bbb", new("BBB")},
		StructPtrField:              new(ComparableStruct{"bbb", new("BBB")}),
		NonComparableStructField:    NonComparableStruct{[]string{"ccc"}},
		NonComparableStructPtrField: new(NonComparableStruct{[]string{"ccc"}}),
		SliceField:                  []string{"ddd"},
		MapField:                    map[string]string{"eee": "eee"},
		ImmutableField:              "fff",
		ImmutablePtrField:           new(ImmutableType("fff")),
	}

	structA2 := structA // dup of A but with different pointer values
	structA2.StringPtrField = new(*structA2.StringPtrField)
	structA2.StructField.StringPtrField = new("BBB")
	structA2.StructPtrField = new(*structA2.StructPtrField)
	structA2.StructPtrField.StringPtrField = new("BBB")
	structA2.NonComparableStructPtrField = new(*structA2.NonComparableStructPtrField)
	structA2.ImmutablePtrField = new(*structA2.ImmutablePtrField)

	structB := Struct{
		StringField:                 "uuu",
		StringPtrField:              new("uuu"),
		StructField:                 ComparableStruct{"vvv", new("VVV")},
		StructPtrField:              new(ComparableStruct{"vvv", new("VVV")}),
		NonComparableStructField:    NonComparableStruct{[]string{"www"}},
		NonComparableStructPtrField: new(NonComparableStruct{[]string{"www"}}),
		SliceField:                  []string{"xxx"},
		MapField:                    map[string]string{"yyy": "yyy"},
		ImmutableField:              "zzz",
		ImmutablePtrField:           new(ImmutableType("zzz")),
	}

	st.Value(&structA).OldValue(&structA).ExpectValid()
	st.Value(&structA).OldValue(&structA2).ExpectValid()
	st.Value(&structA2).OldValue(&structA).ExpectValid()

	st.Value(&structA).OldValue(&structB).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().MatchShortCircuit(), field.ErrorList{
		field.Invalid(field.NewPath("stringField"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("stringPtrField"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("structField"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("structPtrField"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("noncomparableStructField"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("noncomparableStructPtrField"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("sliceField"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("mapField"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("immutableField"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("immutablePtrField"), nil, "").WithOrigin("immutable"),
	}.MarkShortCircuit())
}
