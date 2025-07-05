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
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values
	}).ExpectRegexpsByPath(map[string][]string{
		// All ifOptionDisabled validations should trigger
		"xDisabledField": {"field Struct.XDisabledField"},
		"yDisabledField": {"field Struct.YDisabledField"},
		"xyMixedField":   {"field Struct.XYMixedField/Y"},
	})

	st.Value(&Struct{
		// All zero values
	}).Opts([]string{"FeatureX", "FeatureY"}).ExpectRegexpsByPath(map[string][]string{
		// All ifOptionEnabled validations should trigger
		"xEnabledField": {"field Struct.XEnabledField"},
		"yEnabledField": {"field Struct.YEnabledField"},
		"xyMixedField":  {"field Struct.XYMixedField/X"},
	})

	st.Value(&Struct{
		// All zero values
	}).Opts([]string{"FeatureX"}).ExpectRegexpsByPath(map[string][]string{
		// All ifOptionEnabled validations should trigger
		"xEnabledField":  {"field Struct.XEnabledField"},
		"yDisabledField": {"field Struct.YDisabledField"},
		"xyMixedField": {
			"field Struct.XYMixedField/X",
			"field Struct.XYMixedField/Y"},
	})

	st.Value(&Struct{
		// All zero values
	}).Opts([]string{"FeatureY"}).ExpectRegexpsByPath(map[string][]string{
		// All ifOptionEnabled validations should trigger
		"xDisabledField": {"field Struct.XDisabledField"},
		"yEnabledField":  {"field Struct.YEnabledField"},
	})
}
