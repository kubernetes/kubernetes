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
		field.NotSupported(field.NewPath("enumWithExcludeField"), EnumWithExclude(""), []EnumWithExclude{EnumWithExclude1}),
	})

	st.Value(&Struct{
		Enum0Field:              "",                // no valid value exists
		Enum0PtrField:           ptr.To(Enum0("")), // no valid value exists
		Enum1Field:              E1V1,
		Enum1PtrField:           ptr.To(E1V1),
		Enum2Field:              E2V1,
		Enum2PtrField:           ptr.To(E2V1),
		NotEnumField:            "x",
		NotEnumPtrField:         ptr.To(NotEnum("x")),
		EnumWithExcludeField:    EnumWithExclude1,
		EnumWithExcludePtrField: ptr.To(EnumWithExclude1),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("enum0Field"), Enum0(""), []Enum0{}),
		field.NotSupported(field.NewPath("enum0PtrField"), Enum0(""), []Enum0{}),
	})

	st.Value(&Struct{
		Enum0Field:              "x",                // no valid value exists
		Enum0PtrField:           ptr.To(Enum0("x")), // no valid value exists
		Enum1Field:              "x",
		Enum1PtrField:           ptr.To(Enum1("x")),
		Enum2Field:              "x",
		Enum2PtrField:           ptr.To(Enum2("x")),
		NotEnumField:            "x",
		NotEnumPtrField:         ptr.To(NotEnum("x")),
		EnumWithExcludeField:    "x",
		EnumWithExcludePtrField: ptr.To(EnumWithExclude("x")),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("enum0Field"), Enum0("x"), []Enum0{}),
		field.NotSupported(field.NewPath("enum0PtrField"), Enum0("x"), []Enum0{}),
		field.NotSupported(field.NewPath("enum1Field"), Enum1("x"), []Enum1{E1V1}),
		field.NotSupported(field.NewPath("enum1PtrField"), Enum1("x"), []Enum1{E1V1}),
		field.NotSupported(field.NewPath("enum2Field"), Enum2("x"), []Enum2{E2V1, E2V2}),
		field.NotSupported(field.NewPath("enum2PtrField"), Enum2("x"), []Enum2{E2V1, E2V2}),
		field.NotSupported(field.NewPath("enumWithExcludeField"), EnumWithExclude("x"), []EnumWithExclude{EnumWithExclude1}),
		field.NotSupported(field.NewPath("enumWithExcludePtrField"), EnumWithExclude("x"), []EnumWithExclude{EnumWithExclude1}),
	})

	// Conditional enum tests
	// Scenario 0: Zero value is valid
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "",
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum(""), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD}),
	})

	// Scenario 1: No options (default)
	// Valid values: A, C, D
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("B"), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("E"), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("F"), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).ExpectValid()

	// Scenario 2: FeatureA enabled
	// Valid values: C
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).Opts([]string{"FeatureA"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("A"), []ConditionalEnum{ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).Opts([]string{"FeatureA"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("B"), []ConditionalEnum{ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).Opts([]string{"FeatureA"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("D"), []ConditionalEnum{ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).Opts([]string{"FeatureA"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("E"), []ConditionalEnum{ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).Opts([]string{"FeatureA"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("F"), []ConditionalEnum{ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).Opts([]string{"FeatureA"}).ExpectValid()

	// Scenario 3: FeatureB enabled
	// Valid values: A, B, C
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).Opts([]string{"FeatureB"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("D"), []ConditionalEnum{ConditionalA, ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).Opts([]string{"FeatureB"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("E"), []ConditionalEnum{ConditionalA, ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).Opts([]string{"FeatureB"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("F"), []ConditionalEnum{ConditionalA, ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).Opts([]string{"FeatureB"}).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).Opts([]string{"FeatureB"}).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).Opts([]string{"FeatureB"}).ExpectValid()

	// Scenario 4: FeatureA and FeatureB enabled
	// Valid values: B, C
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).Opts([]string{"FeatureA", "FeatureB"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("A"), []ConditionalEnum{ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).Opts([]string{"FeatureA", "FeatureB"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("D"), []ConditionalEnum{ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).Opts([]string{"FeatureA", "FeatureB"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("E"), []ConditionalEnum{ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).Opts([]string{"FeatureA", "FeatureB"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("F"), []ConditionalEnum{ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).Opts([]string{"FeatureA", "FeatureB"}).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).Opts([]string{"FeatureA", "FeatureB"}).ExpectValid()

	// Scenario 5: FeatureC and FeatureD enabled
	// Valid values: A, C, D, E
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).Opts([]string{"FeatureC", "FeatureD"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("B"), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD, ConditionalE}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).Opts([]string{"FeatureC", "FeatureD"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("F"), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD, ConditionalE}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).Opts([]string{"FeatureC", "FeatureD"}).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).Opts([]string{"FeatureC", "FeatureD"}).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).Opts([]string{"FeatureC", "FeatureD"}).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).Opts([]string{"FeatureC", "FeatureD"}).ExpectValid()

	// Scenario 6: FeatureB and FeatureC enabled
	// Valid values: A, B, C, F
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).Opts([]string{"FeatureB", "FeatureC"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("D"), []ConditionalEnum{ConditionalA, ConditionalB, ConditionalC, ConditionalF}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).Opts([]string{"FeatureB", "FeatureC"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("E"), []ConditionalEnum{ConditionalA, ConditionalB, ConditionalC, ConditionalF}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).Opts([]string{"FeatureB", "FeatureC"}).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).Opts([]string{"FeatureB", "FeatureC"}).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).Opts([]string{"FeatureB", "FeatureC"}).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).Opts([]string{"FeatureB", "FeatureC"}).ExpectValid()
}
