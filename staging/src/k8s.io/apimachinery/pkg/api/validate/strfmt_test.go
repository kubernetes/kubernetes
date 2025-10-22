/*
Copyright 2025 The Kubernetes Authors.

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
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestShortName(t *testing.T) {
	ctx := context.Background()
	fldPath := field.NewPath("test")

	testCases := []struct {
		name     string
		input    string
		wantErrs field.ErrorList
	}{{
		name:     "valid",
		input:    "abc-123",
		wantErrs: nil,
	}, {
		name:  "invalid: empty",
		input: "",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character").WithOrigin("format=k8s-short-name"),
		},
	}, {
		name:  "invalid: too long",
		input: "01234567890123456789012345678901234567890123456789012345678901234",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "01234567890123456789012345678901234567890123456789012345678901234", "must be no more than 63 bytes").WithOrigin("format=k8s-short-name"),
		},
	}, {
		name:  "invalid: starts with dash",
		input: "-abc-123",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "-abc-123", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character").WithOrigin("format=k8s-short-name"),
		},
	}, {
		name:  "invalid: ends with dash",
		input: "abc-123-",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "abc-123-", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character").WithOrigin("format=k8s-short-name"),
		},
	}, {
		name:  "invalid: upper-case",
		input: "ABC-123",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "ABC-123", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character").WithOrigin("format=k8s-short-name"),
		},
	}, {
		name:  "invalid: other chars",
		input: "abc_123",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "abc_123", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character").WithOrigin("format=k8s-short-name"),
		},
	}}

	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailSubstring()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			value := tc.input
			gotErrs := ShortName(ctx, operation.Operation{}, fldPath, &value, nil)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestLongName(t *testing.T) {
	ctx := context.Background()
	fldPath := field.NewPath("test")

	testCases := []struct {
		name     string
		input    string
		wantErrs field.ErrorList
	}{{
		name:     "valid",
		input:    "a.b.c",
		wantErrs: nil,
	}, {
		name:  "invalid: empty",
		input: "",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character").WithOrigin("format=k8s-long-name"),
		},
	}, {
		name:  "invalid: too long",
		input: strings.Repeat("a", 254),
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, strings.Repeat("a", 254), "must be no more than 253 bytes").WithOrigin("format=k8s-long-name"),
		},
	}, {
		name:  "invalid: starts with dash",
		input: "-a.b.c",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "-a.b.c", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character").WithOrigin("format=k8s-long-name"),
		},
	}, {
		name:  "invalid: ends with dash",
		input: "a.b.c-",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "a.b.c-", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character").WithOrigin("format=k8s-long-name"),
		},
	}, {
		name:  "invalid: upper-case",
		input: "A.b.c",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "A.b.c", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character").WithOrigin("format=k8s-long-name"),
		},
	}, {
		name:  "invalid: other chars",
		input: "a_b.c",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "a_b.c", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character").WithOrigin("format=k8s-long-name"),
		},
	}}

	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailSubstring()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			value := tc.input
			gotErrs := LongName(ctx, operation.Operation{}, fldPath, &value, nil)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestLabelKey(t *testing.T) {
	ctx := context.Background()
	fldPath := field.NewPath("test")

	testCases := []struct {
		name     string
		input    string
		wantErrs field.ErrorList
	}{{
		name:     "valid key",
		input:    "app",
		wantErrs: nil,
	}, {
		name:     "valid key with dash",
		input:    "app-name",
		wantErrs: nil,
	}, {
		name:     "valid key with dot",
		input:    "app.name",
		wantErrs: nil,
	}, {
		name:     "valid key with underscore",
		input:    "app_name",
		wantErrs: nil,
	}, {
		name:     "valid key with prefix",
		input:    "example.com/app",
		wantErrs: nil,
	}, {
		name:     "valid key with long prefix",
		input:    strings.Repeat("a", 63) + "." + strings.Repeat("b", 63) + "." + strings.Repeat("c", 63) + "." + strings.Repeat("d", 55) + "/app",
		wantErrs: nil,
	}, {
		name:  "invalid: empty string",
		input: "",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-key"),
		},
	}, {
		name:  "invalid: starts with dash",
		input: "-app",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-key"),
		},
	}, {
		name:  "invalid: ends with dash",
		input: "app-",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-key"),
		},
	}, {
		name:  "invalid: contains invalid characters",
		input: "app^",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-key"),
		},
	}, {
		name:  "invalid: name too long",
		input: strings.Repeat("a", 64),
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-key"),
		},
	}, {
		name:  "invalid: prefix too long",
		input: strings.Repeat("a", 254) + "/app",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-key"),
		},
	}, {
		name:  "invalid: prefix is not a DNS subdomain",
		input: "example-.com/app",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-key"),
		},
	}, {
		name:     "nil value",
		input:    "", // This will be handled by setting value to nil in the test runner
		wantErrs: nil,
	}}

	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var value *string
			if tc.name != "nil value" {
				v := tc.input
				value = &v
			}
			gotErrs := LabelKey(ctx, operation.Operation{}, fldPath, value, nil)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestK8sUUID(t *testing.T) {
	ctx := context.Background()
	fldPath := field.NewPath("test")

	testCases := []struct {
		name     string
		input    string
		wantErrs field.ErrorList
	}{{
		name:     "valid uuid with hyphens",
		input:    "123e4567-e89b-12d3-a456-426614174000",
		wantErrs: nil,
	}, {
		name:  "invalid uuid with hyphens uppercase",
		input: "123E4567-E89B-12D3-A456-426614174000",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "123E4567-E89B-12D3-A456-426614174000", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
		},
	}, {
		name:  "invalid uuid with urn prefix",
		input: "urn:uuid:123e4567-e89b-12d3-a456-426614174000",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "urn:uuid:123e4567-e89b-12d3-a456-426614174000", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
		},
	}, {
		name:  "invalid uuid without hyphens",
		input: "123e4567e89b12d3a456426614174000",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "123e4567e89b12d3a456426614174000", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
		},
	}, {
		name:  "invalid: wrong length",
		input: "123e4567-e89b-12d3-a456-42661417400",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "123e4567-e89b-12d3-a456-42661417400", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
		},
	}, {
		name:  "invalid: wrong characters",
		input: "123e4567-e89b-12d3-a456-42661417400g",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "123e4567-e89b-12d3-a456-42661417400g", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
		},
	}, {
		name:  "invalid: misplaced hyphens",
		input: "123e4567-e89b-12d3-a4564-26614174000",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "123e4567-e89b-12d3-a4564-26614174000", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
		},
	}, {
		name:  "empty string",
		input: "",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
		},
	}, {
		name:  "not a uuid",
		input: "not-a-uuid",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "not-a-uuid", "must be a lowercase UUID in 8-4-4-4-12 format").WithOrigin("format=k8s-uuid"),
		},
	}}

	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByDetailExact()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			value := tc.input
			gotErrs := UUID(ctx, operation.Operation{}, fldPath, &value, nil)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestLabelValue(t *testing.T) {
	ctx := context.Background()
	fldPath := field.NewPath("test")

	testCases := []struct {
		name     string
		input    string
		wantErrs field.ErrorList
	}{{
		name:     "valid value",
		input:    "valid-value",
		wantErrs: nil,
	}, {
		name:     "valid value with dots",
		input:    "valid.value",
		wantErrs: nil,
	}, {
		name:     "valid value with underscores",
		input:    "valid_value",
		wantErrs: nil,
	}, {
		name:     "valid single character value",
		input:    "a",
		wantErrs: nil,
	}, {
		name:     "valid value with numbers",
		input:    "123-abc",
		wantErrs: nil,
	}, {
		name:     "valid uppercase characters",
		input:    "Valid-Value",
		wantErrs: nil,
	}, {
		name:  "invalid: starts with dash",
		input: "-invalid-value",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-value"),
		},
	}, {
		name:  "invalid: ends with dash",
		input: "invalid-value-",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-value"),
		},
	}, {
		name:  "invalid: starts with dot",
		input: ".invalid.value",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-value"),
		},
	}, {
		name:  "invalid: ends with dot",
		input: "invalid.value.",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-value"),
		},
	}, {
		name:  "invalid: starts with underscore",
		input: "_invalid_value",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-value"),
		},
	}, {
		name:  "invalid: ends with underscore",
		input: "invalid_value_",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-value"),
		},
	}, {
		name:  "invalid: contains special characters",
		input: "invalid@value",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-value"),
		},
	}, {
		name:  "invalid: too long",
		input: "a" + strings.Repeat("b", 62) + "c", // 64 characters
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-label-value"),
		},
	}, {
		name:     "valid: max length",
		input:    "a" + strings.Repeat("b", 61) + "c", // 63 characters
		wantErrs: nil,
	}, {
		name:     "valid: empty string",
		input:    "",
		wantErrs: nil,
	}}

	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			value := tc.input
			gotErrs := LabelValue(ctx, operation.Operation{}, fldPath, &value, nil)

			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}

func TestLongNameCaseless(t *testing.T) {
	ctx := context.Background()
	fldPath := field.NewPath("test")

	testCases := []struct {
		name     string
		input    string
		wantErrs field.ErrorList
	}{{
		name:     "valid",
		input:    "A.b.C",
		wantErrs: nil,
	}, {
		name:  "invalid: empty",
		input: "",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "", "an RFC 1123 subdomain must consist of alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character").WithOrigin("format=k8s-long-name-caseless"),
		},
	}, {
		name:  "invalid: too long",
		input: strings.Repeat("a", 254),
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, strings.Repeat("a", 254), "must be no more than 253 bytes").WithOrigin("format=k8s-long-name-caseless"),
		},
	}, {
		name:  "invalid: starts with dash",
		input: "-A.b.C",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, "-A.b.C", "an RFC 1123 subdomain must consist of alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character").WithOrigin("format=k8s-long-name-caseless"),
		},
	},
		{
			name:  "invalid: ends with dash",
			input: "A.b.C-",
			wantErrs: field.ErrorList{
				field.Invalid(fldPath, "A.b.C-", "an RFC 1123 subdomain must consist of alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character").WithOrigin("format=k8s-long-name-caseless"),
			},
		},
		{
			name:  "invalid: other chars",
			input: "A_b.C",
			wantErrs: field.ErrorList{
				field.Invalid(fldPath, "A_b.C", "an RFC 1123 subdomain must consist of alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character").WithOrigin("format=k8s-long-name-caseless"),
			},
		},
	}

	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			value := tc.input
			gotErrs := LongNameCaseless(ctx, operation.Operation{}, fldPath, &value, nil)

			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}
