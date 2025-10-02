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

package field

import (
	"fmt"
	"regexp"
	"strings"
	"testing"
)

func TestErrorMatcher_Matches(t *testing.T) {
	baseErr := func() *Error {
		return &Error{
			Type:     ErrorTypeInvalid,
			Field:    "field",
			BadValue: "value",
			Detail:   "detail",
			Origin:   "origin",
		}
	}

	testCases := []struct {
		name      string
		matcher   ErrorMatcher
		wantedErr func() *Error
		actualErr *Error
		matches   bool
	}{{
		name:      "ByType: match",
		matcher:   ErrorMatcher{}.ByType(),
		wantedErr: baseErr,
		actualErr: &Error{Type: ErrorTypeInvalid},
		matches:   true,
	}, {
		name:      "ByType: no match",
		matcher:   ErrorMatcher{}.ByType(),
		wantedErr: baseErr,
		actualErr: &Error{Type: ErrorTypeRequired},
		matches:   false,
	}, {
		name:      "ByField: match",
		matcher:   ErrorMatcher{}.ByField(),
		wantedErr: baseErr,
		actualErr: &Error{Field: "field"},
		matches:   true,
	}, {
		name:      "ByField: no match",
		matcher:   ErrorMatcher{}.ByField(),
		wantedErr: baseErr,
		actualErr: &Error{Field: "other"},
		matches:   false,
	}, {
		name: "ByFieldNormalized: older API to latest",
		matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
			{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
		}),
		wantedErr: func() *Error {
			e := baseErr()
			e.Field = "f[0].x.a"
			return e
		},
		actualErr: &Error{Field: "f[0].a"},
		matches:   true,
	}, {
		name: "ByFieldNormalized: both latest format",
		matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
			{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
		}),
		wantedErr: func() *Error {
			e := baseErr()
			e.Field = "f[0].x.a"
			return e
		},
		actualErr: &Error{Field: "f[0].x.a"},
		matches:   true,
	}, {
		name: "ByFieldNormalized: different index",
		matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
			{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
		}),
		wantedErr: func() *Error {
			e := baseErr()
			e.Field = "f[0].x.a"
			return e
		},
		actualErr: &Error{Field: "f[1].a"},
		matches:   false,
	}, {
		name: "ByFieldNormalized: multiple patterns",
		matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
			{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
			{Regexp: regexp.MustCompile(`f\[(\d+)\]\.b`), Replacement: "f[$1].x.b"},
		}),
		wantedErr: func() *Error {
			e := baseErr()
			e.Field = "f[2].x.b"
			return e
		},
		actualErr: &Error{Field: "f[2].b"},
		matches:   true,
	}, {
		name: "ByFieldNormalized: no normalization needed",
		matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
			{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
		}),
		wantedErr: func() *Error {
			e := baseErr()
			e.Field = "other.field"
			return e
		},
		actualErr: &Error{Field: "other.field"},
		matches:   true,
	}, {
		name:      "ByValue: match",
		matcher:   ErrorMatcher{}.ByValue(),
		wantedErr: baseErr,
		actualErr: &Error{BadValue: "value"},
		matches:   true,
	}, {
		name:      "ByValue: no match",
		matcher:   ErrorMatcher{}.ByValue(),
		wantedErr: baseErr,
		actualErr: &Error{BadValue: "other"},
		matches:   false,
	}, {
		name:      "ByOrigin: match",
		matcher:   ErrorMatcher{}.ByOrigin(),
		wantedErr: baseErr,
		actualErr: &Error{Origin: "origin"},
		matches:   true,
	}, {
		name:      "ByOrigin: no match",
		matcher:   ErrorMatcher{}.ByOrigin(),
		wantedErr: baseErr,
		actualErr: &Error{Origin: "other"},
		matches:   false,
	}, {
		name:      "ByDetailExact: match",
		matcher:   ErrorMatcher{}.ByDetailExact(),
		wantedErr: baseErr,
		actualErr: &Error{Detail: "detail"},
		matches:   true,
	}, {
		name:      "ByDetailExact: no match",
		matcher:   ErrorMatcher{}.ByDetailExact(),
		wantedErr: baseErr,
		actualErr: &Error{Detail: "other"},
		matches:   false,
	}, {
		name:    "ByDetailSubstring: match empty",
		matcher: ErrorMatcher{}.ByDetailSubstring(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = ""
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   true,
	}, {
		name:    "ByDetailSubstring: match full",
		matcher: ErrorMatcher{}.ByDetailSubstring(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = "is the"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   true,
	}, {
		name:    "ByDetailSubstring: match start",
		matcher: ErrorMatcher{}.ByDetailSubstring(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = "this is"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   true,
	}, {
		name:    "ByDetailSubstring: match middle",
		matcher: ErrorMatcher{}.ByDetailSubstring(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = "is the"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   true,
	}, {
		name:    "ByDetailSubstring: match end",
		matcher: ErrorMatcher{}.ByDetailSubstring(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = "the detail"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   true,
	}, {
		name:    "ByDetailSubstring: no match",
		matcher: ErrorMatcher{}.ByDetailSubstring(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = "is not the"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   false,
	}, {
		name:    "ByDetailRegexp: match empty",
		matcher: ErrorMatcher{}.ByDetailRegexp(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = ".*"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   true,
	}, {
		name:    "ByDetailRegexp: match full",
		matcher: ErrorMatcher{}.ByDetailRegexp(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = "^this is the detail$"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   true,
	}, {
		name:    "ByDetailRegexp: match start",
		matcher: ErrorMatcher{}.ByDetailRegexp(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = "^this is"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   true,
	}, {
		name:    "ByDetailRegexp: match middle",
		matcher: ErrorMatcher{}.ByDetailRegexp(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = "is the"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   true,
	}, {
		name:    "ByDetailRegexp: match end",
		matcher: ErrorMatcher{}.ByDetailRegexp(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = "the detail$"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   true,
	}, {
		name:    "ByDetailRegexp: match parts",
		matcher: ErrorMatcher{}.ByDetailRegexp(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = "^this .* .* detail$"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   true,
	}, {
		name:    "ByDetailRegexp: no match",
		matcher: ErrorMatcher{}.ByDetailRegexp(),
		wantedErr: func() *Error {
			e := baseErr()
			e.Detail = "is not the"
			return e
		},
		actualErr: &Error{Detail: "this is the detail"},
		matches:   false,
	}, {
		name:      "Exactly: match",
		matcher:   ErrorMatcher{}.Exactly(),
		wantedErr: baseErr,
		actualErr: baseErr(),
		matches:   true,
	}, {
		name:      "Exactly: no match (type)",
		matcher:   ErrorMatcher{}.Exactly(),
		wantedErr: baseErr,
		actualErr: &Error{Type: ErrorTypeRequired, Field: "field", BadValue: "value", Detail: "detail", Origin: "origin"},
		matches:   false,
	}, {
		name:      "RequireOriginWhenInvalid: match",
		matcher:   ErrorMatcher{}.ByOrigin().RequireOriginWhenInvalid(),
		wantedErr: baseErr,
		actualErr: &Error{Type: ErrorTypeInvalid, Origin: "origin"},
		matches:   true,
	}, {
		name:      "RequireOriginWhenInvalid: no match (missing origin)",
		matcher:   ErrorMatcher{}.ByOrigin().RequireOriginWhenInvalid(),
		wantedErr: baseErr,
		actualErr: &Error{Type: ErrorTypeInvalid},
		matches:   false,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.matcher.Matches(tc.wantedErr(), tc.actualErr) != tc.matches {
				t.Errorf("Matches() = %v, want %v", !tc.matches, tc.matches)
			}
		})
	}
}

// fakeTestIntf is used to test the testing support.
type fakeTestIntf struct {
	errs []string
}

var _ TestIntf = &fakeTestIntf{}

func (*fakeTestIntf) Helper() {}

func (ft *fakeTestIntf) Errorf(format string, args ...any) {
	ft.errs = append(ft.errs, fmt.Sprintf(format, args...))
}

func TestErrorMatcher_Test(t *testing.T) {
	testCases := []struct {
		name           string
		matcher        ErrorMatcher
		want           ErrorList
		got            ErrorList
		expectedErrors []string
		expectedLogs   []string
	}{{
		name:    "no origin: perfect match",
		matcher: ErrorMatcher{}.ByField(),
		want:    ErrorList{Invalid(NewPath("f"), nil, "")},
		got:     ErrorList{Invalid(NewPath("f"), "v", "d")},
	}, {
		name:           "no origin: got too few errors",
		matcher:        ErrorMatcher{}.ByField(),
		want:           ErrorList{Invalid(NewPath("f"), nil, "")},
		got:            ErrorList{},
		expectedErrors: []string{"expected an error matching:"},
	}, {
		name:           "no origin: got too many errors",
		matcher:        ErrorMatcher{}.ByField(),
		want:           ErrorList{},
		got:            ErrorList{Invalid(NewPath("f"), "v", "d")},
		expectedErrors: []string{"unmatched error:"},
	}, {
		name:           "no origin: got wrong errors",
		matcher:        ErrorMatcher{}.ByField(),
		want:           ErrorList{Invalid(NewPath("f1"), nil, "")},
		got:            ErrorList{Invalid(NewPath("f2"), "v", "d")},
		expectedErrors: []string{"expected an error matching:", "unmatched error:"},
	}, {
		name: "with normalization: older API to latest",
		matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
			{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
		}),
		want: ErrorList{Invalid(NewPath("f").Index(0).Child("x", "a"), nil, "")},
		got:  ErrorList{Invalid(NewPath("f").Index(0).Child("a"), "v", "d")},
	}, {
		name: "with normalization: both latest",
		matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
			{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
		}),
		want: ErrorList{Invalid(NewPath("f").Index(0).Child("x", "a"), nil, "")},
		got:  ErrorList{Invalid(NewPath("f").Index(0).Child("x", "a"), "v", "d")},
	}, {
		name: "with normalization: multiple",
		matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
			{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
			{Regexp: regexp.MustCompile(`f\[(\d+)\]\.b`), Replacement: "f[$1].x.b"},
		}),
		want: ErrorList{
			Invalid(NewPath("f").Index(0).Child("x", "a"), nil, ""),
			Invalid(NewPath("f").Index(1).Child("x", "b"), nil, ""),
		},
		got: ErrorList{
			Invalid(NewPath("f").Index(0).Child("a"), "v1", "d1"),
			Invalid(NewPath("f").Index(1).Child("b"), "v2", "d2"),
		},
	}, {
		name: "with normalization: no match",
		matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
			{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
		}),
		want:           ErrorList{Invalid(NewPath("f").Index(0).Child("x", "a"), nil, "")},
		got:            ErrorList{Invalid(NewPath("f").Index(1).Child("a"), "v", "d")},
		expectedErrors: []string{"expected an error matching:", "unmatched error:"},
	}, {
		name:    "with origin: single match",
		matcher: ErrorMatcher{}.ByField().ByOrigin(),
		want:    ErrorList{Invalid(NewPath("f"), nil, "").WithOrigin("o")},
		got:     ErrorList{Invalid(NewPath("f"), "v", "d").WithOrigin("o")},
	}, {
		name:    "with origin: multiple matches, different details",
		matcher: ErrorMatcher{}.ByField().ByOrigin(),
		want: ErrorList{
			Invalid(NewPath("f1"), nil, "").WithOrigin("o"),
			Invalid(NewPath("f2"), nil, "").WithOrigin("o"),
		},
		got: ErrorList{
			Invalid(NewPath("f1"), "v", "d1").WithOrigin("o"),
			Invalid(NewPath("f2"), "v", "d1").WithOrigin("o"),
			Invalid(NewPath("f1"), "v", "d2").WithOrigin("o"),
			Invalid(NewPath("f2"), "v", "d2").WithOrigin("o"),
		},
	}, {
		name:    "with origin: multiple matches, same exact error",
		matcher: ErrorMatcher{}.ByField().ByOrigin(),
		want: ErrorList{
			Invalid(NewPath("f1"), nil, "").WithOrigin("o"),
			Invalid(NewPath("f2"), nil, "").WithOrigin("o"),
		},
		got: ErrorList{
			Invalid(NewPath("f1"), "v", "d").WithOrigin("o"),
			Invalid(NewPath("f1"), "v", "d").WithOrigin("o"),
			Invalid(NewPath("f2"), "v", "d").WithOrigin("o"),
			Invalid(NewPath("f2"), "v", "d").WithOrigin("o"),
		},
		expectedErrors: []string{"exact duplicate error:", "exact duplicate error:"},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fakeT := &fakeTestIntf{}
			tc.matcher.Test(fakeT, tc.want, tc.got)
			if want, got := len(tc.expectedErrors), len(fakeT.errs); got != want {
				if got == 0 {
					t.Errorf("expected %d errors, got %d", want, got)
				} else {
					q := make([]string, len(fakeT.errs))
					for i, err := range fakeT.errs {
						q[i] = fmt.Sprintf("%q", err)
					}
					t.Errorf("expected %d errors, got %d:\n%s", want, got, strings.Join(q, "\n"))
				}
			} else {
				for i := range tc.expectedErrors {
					if !strings.HasPrefix(fakeT.errs[i], tc.expectedErrors[i]) {
						t.Errorf("error %d: expected prefix %q, got %q", i, tc.expectedErrors[i], fakeT.errs[i])
					}
				}
			}
		})
	}
}

func TestErrorMatcher_Render(t *testing.T) {
	testCases := []struct {
		name     string
		matcher  ErrorMatcher
		err      *Error
		expected string
	}{
		{
			name:     "empty matcher",
			matcher:  ErrorMatcher{},
			err:      Invalid(NewPath("field"), "value", "detail"),
			expected: "{}",
		},
		{
			name:     "single field - type",
			matcher:  ErrorMatcher{}.ByType(),
			err:      Invalid(NewPath("field"), "value", "detail"),
			expected: `{Type="Invalid value"}`,
		},
		{
			name:     "single field - value with string",
			matcher:  ErrorMatcher{}.ByValue(),
			err:      Invalid(NewPath("field"), "string_value", "detail"),
			expected: `{Value="string_value"}`,
		},
		{
			name:     "single field - value with nil",
			matcher:  ErrorMatcher{}.ByValue(),
			err:      Invalid(NewPath("field"), nil, "detail"),
			expected: `{Value=<nil>}`,
		},
		{
			name:     "multiple fields",
			matcher:  ErrorMatcher{}.ByType().ByField().ByValue(),
			err:      Invalid(NewPath("field"), "value", "detail"),
			expected: `{Type="Invalid value", Field="field", Value="value"}`,
		},
		{
			name:     "all fields",
			matcher:  ErrorMatcher{}.ByType().ByField().ByValue().ByOrigin().ByDetailExact(),
			err:      Invalid(NewPath("field"), "value", "detail").WithOrigin("origin"),
			expected: `{Type="Invalid value", Field="field", Value="value", Origin="origin", Detail="detail"}`,
		},
		{
			name: "with normalization: normalized",
			matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
				{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
			}),
			err:      Invalid(NewPath("f").Index(0).Child("a"), "value", "detail"),
			expected: `{Field="f[0].x.a" (aka "f[0].a")}`,
		},
		{
			name: "with normalization: no normalization",
			matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
				{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
			}),
			err:      Invalid(NewPath("other", "field"), "value", "detail"),
			expected: `{Field="other.field"}`,
		},
		{
			name: "with normalization: already normalized",
			matcher: ErrorMatcher{}.ByFieldNormalized([]NormalizationRule{
				{Regexp: regexp.MustCompile(`f\[(\d+)\]\.a`), Replacement: "f[$1].x.a"},
			}),
			err:      Invalid(NewPath("f").Index(0).Child("x", "a"), "value", "detail"),
			expected: `{Field="f[0].x.a"}`,
		},
		{
			name:     "requireOriginWhenInvalid with origin",
			matcher:  ErrorMatcher{}.ByOrigin().RequireOriginWhenInvalid(),
			err:      Invalid(NewPath("field"), "value", "detail").WithOrigin("origin"),
			expected: `{Origin="origin"}`,
		},
		{
			name:     "different error types",
			matcher:  ErrorMatcher{}.ByType().ByValue(),
			err:      Required(NewPath("field"), "detail"),
			expected: `{Type="Required value", Value=""}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := tc.matcher.Render(tc.err)
			if result != tc.expected {
				t.Errorf("Render() = %v, want %v", result, tc.expected)
			}
		})
	}
}
