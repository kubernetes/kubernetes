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
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestStructWithModeSetByServer(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Mode = "Auto"
	// AutoSetField is setByServer and optional. When nil, Required error is triggered.
	st.Value(&StructWithModeSetByServer{Mode: "Auto"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("autoSetField"), ""),
	})

	// When AutoSetField is provided in Auto mode, it is valid.
	val := "val"
	st.Value(&StructWithModeSetByServer{Mode: "Auto", AutoSetField: &val}).ExpectValid()

	// ManualField is unlisted for Auto mode, so implicitly forbidden.
	st.Value(&StructWithModeSetByServer{Mode: "Auto", AutoSetField: &val, ManualField: &val}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("manualField"), ""),
	})

	// Mode = "Manual"
	// ManualField has ifMode("Manual") setByServer and forbidden.
	// When nil, setByServer triggers Required error.
	st.Value(&StructWithModeSetByServer{Mode: "Manual"}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("manualField"), ""),
	})

	// When ManualField is provided in Manual mode, it is valid.
	st.Value(&StructWithModeSetByServer{Mode: "Manual", ManualField: &val}).ExpectValid()

	// AutoSetField is unlisted for Manual mode, so implicitly forbidden. ManualField when nil triggers Required.
	st.Value(&StructWithModeSetByServer{Mode: "Manual", AutoSetField: &val}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("manualField"), ""),
		field.Forbidden(field.NewPath("autoSetField"), ""),
	})
}
