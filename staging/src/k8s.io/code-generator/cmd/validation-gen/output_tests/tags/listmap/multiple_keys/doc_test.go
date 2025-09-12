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

package multiplekeys

import (
	"testing"

	field "k8s.io/apimachinery/pkg/util/validation/field"
)

func TestUniqueness(t *testing.T) {
	// TODO: enable this once we have a way to either opt-out from this validation
	// or settle the decision on how to handle the ratcheting cases.
	/*
		st := localSchemeBuilder.Test(t)

		st.Value(&Struct{
			ListField: []OtherStruct{
				{"key1", 1, "one"}, // unique
				{"key2", 2, "two"}, // dup
				{"key2", 2, "two"},
			},
			ListTypedefField: []OtherTypedefStruct{
				{"key1", 1, "one"}, // unique
				{"key2", 2, "two"}, // dup
				{"key2", 2, "two"},
			},
			TypedefField: ListType{
				{"key1", 1, "one"}, // unique
				{"key2", 2, "two"}, // dup
				{"key2", 2, "two"},
			},
		}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
			field.Duplicate(field.NewPath("listField").Index(2), nil),
			field.Duplicate(field.NewPath("listTypedefField").Index(2), nil),
			field.Duplicate(field.NewPath("typedefField").Index(2), nil),
		})
	*/
}

func TestUpdateCorrelation(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	structA1 := Struct{
		ListField: []OtherStruct{
			{"key1", 1, "one"},
			{"key2", 2, "two"},
		},
		ListTypedefField: []OtherTypedefStruct{
			{"key1", 1, "one"},
			{"key2", 2, "two"},
		},
		TypedefField: ListType{
			{"key1", 1, "one"},
			{"key2", 2, "two"},
		},
	}

	// Same data, different order.
	structA2 := Struct{
		ListField: []OtherStruct{
			{"key2", 2, "two"},
			{"key1", 1, "one"},
		},
		ListTypedefField: []OtherTypedefStruct{
			{"key2", 2, "two"},
			{"key1", 1, "one"},
		},
		TypedefField: ListType{
			{"key2", 2, "two"},
			{"key1", 1, "one"},
		},
	}

	// Different data.
	structB := Struct{
		ListField: []OtherStruct{
			{"key3", 3, "THREE"},
			{"key1", 1, "ONE"},
			{"key2", 2, "TWO"},
		},
		ListTypedefField: []OtherTypedefStruct{
			{"key3", 3, "THREE"},
			{"key1", 1, "ONE"},
			{"key2", 2, "TWO"},
		},
		TypedefField: ListType{
			{"key3", 3, "THREE"},
			{"key1", 1, "ONE"},
			{"key2", 2, "TWO"},
		},
	}

	st.Value(&structA1).OldValue(&structA2).ExpectValid()

	st.Value(&structA2).OldValue(&structA1).ExpectValid()

	st.Value(&structA1).OldValue(&structB).ExpectInvalid(
		field.Forbidden(field.NewPath("listField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("listField").Index(1), "field is immutable"),
		field.Forbidden(field.NewPath("listTypedefField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("listTypedefField").Index(1), "field is immutable"),
		field.Forbidden(field.NewPath("typedefField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("typedefField").Index(1), "field is immutable"),
	)

	st.Value(&structB).OldValue(&structA1).ExpectInvalid(
		field.Forbidden(field.NewPath("listField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("listField").Index(1), "field is immutable"),
		field.Forbidden(field.NewPath("listField").Index(2), "field is immutable"),
		field.Forbidden(field.NewPath("listTypedefField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("listTypedefField").Index(1), "field is immutable"),
		field.Forbidden(field.NewPath("listTypedefField").Index(2), "field is immutable"),
		field.Forbidden(field.NewPath("typedefField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("typedefField").Index(1), "field is immutable"),
		field.Forbidden(field.NewPath("typedefField").Index(2), "field is immutable"),
	)
}

func TestRatcheting(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	struct1 := Struct{
		ListComparableField: []OtherStruct{
			{"key1", 1, "one"},
			{"key2", 2, "two"},
		},
		ListNonComparableField: []NonComparableStruct{
			{"key1", 1, []string{"one"}},
			{"key2", 2, []string{"two"}},
		},
	}

	// Same data, different order.
	struct2 := Struct{
		ListComparableField: []OtherStruct{
			{"key2", 2, "two"},
			{"key1", 1, "one"},
		},
		ListNonComparableField: []NonComparableStruct{
			{"key2", 2, []string{"two"}},
			{"key1", 1, []string{"one"}},
		},
	}

	st.Value(&struct1).ExpectValidateFalseByPath(map[string][]string{
		"listComparableField[0]":    {"field Struct.ListComparableField[*]"},
		"listComparableField[1]":    {"field Struct.ListComparableField[*]"},
		"listNonComparableField[0]": {"field Struct.ListNonComparableField[*]"},
		"listNonComparableField[1]": {"field Struct.ListNonComparableField[*]"},
	})
	st.Value(&struct1).OldValue(&struct2).ExpectValid()
	st.Value(&struct2).OldValue(&struct1).ExpectValid()
}
