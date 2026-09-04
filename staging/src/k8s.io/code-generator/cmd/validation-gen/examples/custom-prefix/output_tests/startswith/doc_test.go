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

package startswith

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Unset optional fields are not validated.
	st.Value(&Struct{}).Opts(map[string]bool{"Feature": true}).ExpectValid()

	st.Value(&Struct{
		StringField:    "abcdef",
		StringPtrField: new("abc"),
		SliceField:     []string{"abc", "abcd"},
		GatedField:     "abc",
		TypedefField:   "abc",
	}).Opts(map[string]bool{"Feature": true}).ExpectValid()

	st.Value(&Struct{
		StringField:    "xabc",
		StringPtrField: new(""),
		SliceField:     []string{"abc", "xyz"},
		GatedField:     "xyz",
		TypedefField:   "ab",
	}).Opts(map[string]bool{"Feature": true}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("stringField"), nil, "").WithOrigin("startsWith"),
		field.Invalid(field.NewPath("stringPtrField"), nil, "").WithOrigin("startsWith"),
		field.Invalid(field.NewPath("sliceField").Index(1), nil, "").WithOrigin("startsWith"),
		field.Invalid(field.NewPath("gatedField"), nil, "").WithOrigin("startsWith"),
		field.Invalid(field.NewPath("typedefField"), nil, "").WithOrigin("startsWith"),
	})

	// The gated validation is skipped when the option is disabled.
	st.Value(&Struct{
		StringField:  "abc",
		GatedField:   "xyz",
		TypedefField: "abc",
	}).Opts(map[string]bool{"Feature": false}).ExpectValid()

	// Unchanged invalid values ratchet through on update.
	st.Value(&Struct{
		StringField:  "xabc",
		TypedefField: "abc",
	}).OldValue(&Struct{
		StringField:  "xabc",
		TypedefField: "abc",
	}).Opts(map[string]bool{"Feature": true}).ExpectValid()
}
