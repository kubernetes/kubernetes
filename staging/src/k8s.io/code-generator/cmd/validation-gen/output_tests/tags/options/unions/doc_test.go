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

package unions

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	"testing"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values
	}).ExpectValid()

	st.Value(&Struct{
		UnionField: Union{
			Discriminator:  "xEnabledField",
			XEnabledField:  "foo",
			XDisabledField: "bar", // Invalid since XEnabledField is the discriminator
		},
		ZeroOrOneOfField: ZeroOrOneOf{
			XEnabledField:  "foo",
			XDisabledField: "bar",
		},
		ZeroOrOneOfItem: []Task{
			{Name: "succeeded"},
			{Name: "failed"},
		},
	}).Opts([]string{"FeatureX"}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Invalid(field.NewPath("unionField", "xDisabledField"), "", "").WithOrigin("union"),
			field.Invalid(field.NewPath("unionField", "xEnabledField"), "", "").WithOrigin("union"),
			field.Invalid(field.NewPath("zeroOrOneOfField"), "{XEnabledField:\"foo\", XDisabledField:\"bar\"}", "").WithOrigin("zeroOrOneOf"),
			field.Invalid(field.NewPath("zeroOrOneOfItem"), "[{Name:\"succeeded\", State:\"\"} {Name:\"failed\", State:\"\"}]", "").WithOrigin("zeroOrOneOf"),
		},
	)

	st.Value(&Struct{
		UnionField: Union{
			Discriminator:  "xEnabledField",
			XEnabledField:  "foo",
			XDisabledField: "bar",
		},
		ZeroOrOneOfField: ZeroOrOneOf{
			XEnabledField:  "foo",
			XDisabledField: "bar",
		},
		ZeroOrOneOfItem: []Task{
			{Name: "succeeded"},
			{Name: "failed"},
		},
	}).ExpectValid()

	st.Value(&Struct{
		UnionFieldDisabled: UnionDisabled{
			Discriminator:  "xEnabledField",
			XEnabledField:  "foo",
			XDisabledField: "bar", // Invalid since XEnabledField is the discriminator
		},
		ZeroOrOneOfFieldDisabled: ZeroOrOneOfDisabled{
			XEnabledField:  "foo",
			XDisabledField: "bar",
		},
		ZeroOrOneOfItemDisabled: []Task{
			{Name: "succeeded"},
			{Name: "failed"},
		},
	}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Invalid(field.NewPath("unionFieldDisabled", "xDisabledField"), "", "").WithOrigin("union"),
			field.Invalid(field.NewPath("unionFieldDisabled", "xEnabledField"), "", "").WithOrigin("union"),
			field.Invalid(field.NewPath("zeroOrOneOfFieldDisabled"), "{XEnabledField:\"foo\", XDisabledField:\"bar\"}", "").WithOrigin("zeroOrOneOf"),
			field.Invalid(field.NewPath("zeroOrOneOfItemDisabled"), "[{Name:\"succeeded\", State:\"\"} {Name:\"failed\", State:\"\"}]", "").WithOrigin("zeroOrOneOf"),
		},
	)

	st.Value(&Struct{
		UnionFieldDisabled: UnionDisabled{
			Discriminator:  "xEnabledField",
			XEnabledField:  "foo",
			XDisabledField: "bar",
		},
		ZeroOrOneOfFieldDisabled: ZeroOrOneOfDisabled{
			XEnabledField:  "foo",
			XDisabledField: "bar",
		},
		ZeroOrOneOfItemDisabled: []Task{
			{Name: "succeeded"},
			{Name: "failed"},
		},
	}).Opts([]string{"FeatureX"}).ExpectValid()
}
