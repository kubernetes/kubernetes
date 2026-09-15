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

package validate

import (
	"context"
	"regexp"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestMatches(t *testing.T) {
	re := regexp.MustCompile(`^[a-z]+://[a-z.]+$`)
	const message = "must be a URI"
	const origin = "format=project-uri"

	invalid := func(value string) field.ErrorList {
		return field.ErrorList{
			field.Invalid(field.NewPath("fldpath"), value, content.RegexError(message, re.String())).WithOrigin(origin),
		}
	}

	cases := []struct {
		name     string
		value    *string
		wantErrs field.ErrorList
	}{{
		name:  "matches",
		value: new("https://example.com"),
	}, {
		name:     "does not match",
		value:    new("example.com"),
		wantErrs: invalid("example.com"),
	}, {
		name:     "empty",
		value:    new(""),
		wantErrs: invalid(""),
	}, {
		// Matches does not anchor, so this anchored pattern must reject a
		// value that merely contains a match.
		name:     "unanchored substring",
		value:    new("see https://example.com now"),
		wantErrs: invalid("see https://example.com now"),
	}, {
		// An unset optional field. Whether a value is required is checked
		// separately, not by a format.
		name:  "nil",
		value: nil,
	}}

	matcher := field.ErrorMatcher{}.ByOrigin().ByDetailSubstring().ByField().ByType()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotErrs := Matches(context.Background(), operation.Operation{}, field.NewPath("fldpath"), tc.value, nil,
				re, message, origin)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}
