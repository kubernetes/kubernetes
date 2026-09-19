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

package setbyserver

import (
	"errors"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Case 1: FeatureX=false, FeatureY=false
	// Active setByServer rules:
	// - XDisabledField (ifDisabled(FeatureX))
	// - YDisabledField (ifDisabled(FeatureY))
	// - SubStruct.YDisabledSubfield (ifDisabled(FeatureY))
	st.Value(&Struct{
		SubStruct: &Submarker{},
	}).Opts(map[string]bool{"FeatureX": false, "FeatureY": false}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("xDisabledField"), ""),
		field.Required(field.NewPath("yDisabledField"), ""),
		field.Required(field.NewPath("subStruct", "yDisabledSubfield"), ""),
	})

	// Case 2: FeatureX=true, FeatureY=true
	// Active setByServer rules:
	// - XEnabledField (ifEnabled(FeatureX))
	// - YEnabledField (ifEnabled(FeatureY) setByServer is active; when nil it triggers required)
	// - SubStruct.XEnabledSubfield (ifEnabled(FeatureX))
	st.Value(&Struct{
		SubStruct: &Submarker{},
	}).Opts(map[string]bool{"FeatureX": true, "FeatureY": true}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("xEnabledField"), ""),
		field.Required(field.NewPath("yEnabledField"), ""),
		field.Required(field.NewPath("subStruct", "xEnabledSubfield"), ""),
	})

	// Case 2b: FeatureX=true, FeatureY=true with YEnabledField provided -> valid for YEnabledField
	st.Value(&Struct{
		YEnabledField: new(string),
		SubStruct:     &Submarker{},
	}).Opts(map[string]bool{"FeatureX": true, "FeatureY": true}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("xEnabledField"), ""),
		field.Required(field.NewPath("subStruct", "xEnabledSubfield"), ""),
	})

	// Case 3: FeatureX=true, FeatureY=false
	// Active setByServer rules:
	// - XEnabledField (ifEnabled(FeatureX))
	// - YDisabledField (ifDisabled(FeatureY))
	// - SubStruct.XEnabledSubfield & SubStruct.YDisabledSubfield
	st.Value(&Struct{
		SubStruct: &Submarker{},
	}).Opts(map[string]bool{"FeatureX": true, "FeatureY": false}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("xEnabledField"), ""),
		field.Required(field.NewPath("yDisabledField"), ""),
		field.Required(field.NewPath("subStruct", "xEnabledSubfield"), ""),
		field.Required(field.NewPath("subStruct", "yDisabledSubfield"), ""),
	})

	// Case 4: FeatureX=false, FeatureY=true
	// Active setByServer rules:
	// - XDisabledField (ifDisabled(FeatureX))
	// - YEnabledField (ifEnabled(FeatureY) - setByServer active and not forbidden)
	st.Value(&Struct{}).Opts(map[string]bool{"FeatureX": false, "FeatureY": true}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("xDisabledField"), ""),
		field.Required(field.NewPath("yEnabledField"), ""),
	})

	// Case 5: All required fields populated for active feature flags (FeatureX=true, FeatureY=false)
	st.Value(&Struct{
		XEnabledField:  "val1",
		YDisabledField: new(string),
		SubStruct: &Submarker{
			XEnabledSubfield:  "sub1",
			YDisabledSubfield: "sub2",
		},
	}).Opts(map[string]bool{"FeatureX": true, "FeatureY": false}).ExpectValid()

	// Case 6: No options declared: every referenced option is undeclared.
	internal := func(p string) *field.Error {
		return field.InternalError(field.NewPath(p), errors.New(""))
	}
	st.Value(&Struct{
		SubStruct: &Submarker{},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		internal("xEnabledField"),
		internal("xDisabledField"),
		internal("yEnabledField"), // ifEnabled(FeatureY)
		internal("yDisabledField"),
		internal("subStruct"), // ifDisabled(FeatureY)
		internal("subStruct"), // ifEnabled(FeatureX)
	})
}
