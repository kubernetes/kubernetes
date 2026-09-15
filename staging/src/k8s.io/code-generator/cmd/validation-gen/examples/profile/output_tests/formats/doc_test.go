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

package formats

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)
	opts := map[string]bool{"Feature": true}

	// Unset optional fields are not validated.
	st.Value(&Struct{}).Opts(opts).ExpectValid()

	st.Value(&Struct{
		RegexField:      "https://example.com/a",
		OtherRegexField: "git+ssh://example.com/repo",
		DigestField:     "deadbeef",
		SliceField:      []string{"https://example.com", "ftp://example.com/x"},
		MapField:        map[string]string{"cafef00d": "x"},
		GatedField:      "https://example.com",
		TypedefField:    "https://example.com",
		BuiltInField:    "a-name",
	}).Opts(opts).ExpectValid()

	st.Value(&Struct{
		RegexField:      "example.com",
		OtherRegexField: "HTTPS://example.com",
		DigestField:     "DEADBEEF",
		SliceField:      []string{"https://example.com", "nope"},
		MapField:        map[string]string{"nope": "x"},
		GatedField:      "nope",
		TypedefField:    "nope",
		BuiltInField:    "Not A Name",
	}).Opts(opts).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("regexField"), nil, "").WithOrigin("format=example-uri"),
		field.Invalid(field.NewPath("otherRegexField"), nil, "").WithOrigin("format=example-uri"),
		field.Invalid(field.NewPath("digestField"), nil, "").WithOrigin("format=example-hex-digest"),
		field.Invalid(field.NewPath("sliceField").Index(1), nil, "").WithOrigin("format=example-uri"),
		// eachKey reports at the map itself, not under the offending key.
		field.Invalid(field.NewPath("mapField"), nil, "").WithOrigin("format=example-hex-digest"),
		field.Invalid(field.NewPath("gatedField"), nil, "").WithOrigin("format=example-uri"),
		field.Invalid(field.NewPath("typedefField"), nil, "").WithOrigin("format=example-uri"),
		field.Invalid(field.NewPath("builtInField"), nil, "").WithOrigin("format=k8s-short-name"),
	})

	// The gated validation is skipped when the option is disabled, which is
	// what a profile format has to support to be worth preferring over a
	// hand-written customValidation.
	st.Value(&Struct{GatedField: "nope"}).Opts(map[string]bool{"Feature": false}).ExpectValid()
}

// TestRegexErrorMessage checks what a user of the API actually sees. A pattern
// is not an explanation, so the message from the profile has to reach the
// error, alongside the pattern rather than instead of it.
func TestRegexErrorMessage(t *testing.T) {
	errs := Validate_Struct(t.Context(), operation.Operation{Type: operation.Create}, nil,
		&Struct{RegexField: "example.com"}, nil)
	if len(errs) != 1 {
		t.Fatalf("got %v, want one error", errs)
	}
	for _, want := range []string{
		"must be an absolute URI with a scheme",
		`^[a-z][a-z0-9+.-]*://[^\s]+$`,
	} {
		if !strings.Contains(errs[0].Detail, want) {
			t.Errorf("error detail %q does not contain %q", errs[0].Detail, want)
		}
	}

	errs = Validate_Struct(t.Context(), operation.Operation{Type: operation.Create}, nil,
		&Struct{DigestField: "DEADBEEF"}, nil)
	if len(errs) != 1 {
		t.Fatalf("got %v, want one error", errs)
	}
	if !strings.Contains(errs[0].Detail, "must be a lower-case hexadecimal digest") {
		t.Errorf("error detail %q does not contain the message from the profile", errs[0].Detail)
	}
}
