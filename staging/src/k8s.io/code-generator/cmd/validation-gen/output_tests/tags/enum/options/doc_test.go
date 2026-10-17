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

package options

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// opts declares all options this type references, enabling the named ones.
	opts := func(enabled ...string) map[string]bool {
		m := map[string]bool{"FeatureA": false, "FeatureB": false, "FeatureC": false, "FeatureD": false}
		for _, e := range enabled {
			m[e] = true
		}
		return m
	}

	st.Value(&ConditionalStruct{
		ConditionalEnumField: "",
	}).Opts(opts()).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum(""), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD}),
	})

	// Scenario 1: No options (default)
	// Valid values: A, C, D
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).Opts(opts()).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("B"), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).Opts(opts()).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("E"), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).Opts(opts()).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("F"), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).Opts(opts()).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).Opts(opts()).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).Opts(opts()).ExpectValid()

	// Scenario 2: FeatureA enabled
	// Valid values: C
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).Opts(opts("FeatureA")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("A"), []ConditionalEnum{ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).Opts(opts("FeatureA")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("B"), []ConditionalEnum{ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).Opts(opts("FeatureA")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("D"), []ConditionalEnum{ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).Opts(opts("FeatureA")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("E"), []ConditionalEnum{ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).Opts(opts("FeatureA")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("F"), []ConditionalEnum{ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).Opts(opts("FeatureA")).ExpectValid()

	// Scenario 3: FeatureB enabled
	// Valid values: A, B, C
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).Opts(opts("FeatureB")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("D"), []ConditionalEnum{ConditionalA, ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).Opts(opts("FeatureB")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("E"), []ConditionalEnum{ConditionalA, ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).Opts(opts("FeatureB")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("F"), []ConditionalEnum{ConditionalA, ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).Opts(opts("FeatureB")).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).Opts(opts("FeatureB")).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).Opts(opts("FeatureB")).ExpectValid()

	// Scenario 4: FeatureA and FeatureB enabled
	// Valid values: B, C
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).Opts(opts("FeatureA", "FeatureB")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("A"), []ConditionalEnum{ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).Opts(opts("FeatureA", "FeatureB")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("D"), []ConditionalEnum{ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).Opts(opts("FeatureA", "FeatureB")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("E"), []ConditionalEnum{ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).Opts(opts("FeatureA", "FeatureB")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("F"), []ConditionalEnum{ConditionalB, ConditionalC}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).Opts(opts("FeatureA", "FeatureB")).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).Opts(opts("FeatureA", "FeatureB")).ExpectValid()

	// Scenario 5: FeatureC and FeatureD enabled
	// Valid values: A, C, D, E
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).Opts(opts("FeatureC", "FeatureD")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("B"), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD, ConditionalE}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).Opts(opts("FeatureC", "FeatureD")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("F"), []ConditionalEnum{ConditionalA, ConditionalC, ConditionalD, ConditionalE}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).Opts(opts("FeatureC", "FeatureD")).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).Opts(opts("FeatureC", "FeatureD")).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).Opts(opts("FeatureC", "FeatureD")).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).Opts(opts("FeatureC", "FeatureD")).ExpectValid()

	// Scenario 6: FeatureB and FeatureC enabled
	// Valid values: A, B, C, F
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "D",
	}).Opts(opts("FeatureB", "FeatureC")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("D"), []ConditionalEnum{ConditionalA, ConditionalB, ConditionalC, ConditionalF}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "E",
	}).Opts(opts("FeatureB", "FeatureC")).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.NotSupported(field.NewPath("conditionalEnumField"), ConditionalEnum("E"), []ConditionalEnum{ConditionalA, ConditionalB, ConditionalC, ConditionalF}),
	})
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "A",
	}).Opts(opts("FeatureB", "FeatureC")).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "B",
	}).Opts(opts("FeatureB", "FeatureC")).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "C",
	}).Opts(opts("FeatureB", "FeatureC")).ExpectValid()
	st.Value(&ConditionalStruct{
		ConditionalEnumField: "F",
	}).Opts(opts("FeatureB", "FeatureC")).ExpectValid()
}
