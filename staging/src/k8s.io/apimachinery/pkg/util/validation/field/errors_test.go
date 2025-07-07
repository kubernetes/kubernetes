/*
Copyright 2014 The Kubernetes Authors.

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
	"reflect"
	"testing"
	"time"

	"k8s.io/utils/ptr"
)

func TestMakeFuncs(t *testing.T) {
	testCases := []struct {
		fn       func() *Error
		expected ErrorType
	}{
		{
			func() *Error { return Invalid(NewPath("f"), "v", "d") },
			ErrorTypeInvalid,
		},
		{
			func() *Error { return NotSupported[string](NewPath("f"), "v", nil) },
			ErrorTypeNotSupported,
		},
		{
			func() *Error { return Duplicate(NewPath("f"), "v") },
			ErrorTypeDuplicate,
		},
		{
			func() *Error { return NotFound(NewPath("f"), "v") },
			ErrorTypeNotFound,
		},
		{
			func() *Error { return Required(NewPath("f"), "d") },
			ErrorTypeRequired,
		},
		{
			func() *Error { return InternalError(NewPath("f"), fmt.Errorf("e")) },
			ErrorTypeInternal,
		},
	}

	for _, testCase := range testCases {
		err := testCase.fn()
		if err.Type != testCase.expected {
			t.Errorf("expected Type %q, got %q", testCase.expected, err.Type)
		}
	}
}

func TestToAggregate(t *testing.T) {
	testCases := struct {
		ErrList         []ErrorList
		NumExpectedErrs []int
	}{
		[]ErrorList{
			nil,
			{},
			{Invalid(NewPath("f"), "v", "d")},
			{Invalid(NewPath("f"), "v", "d"), Invalid(NewPath("f"), "v", "d")},
			{Invalid(NewPath("f"), "v", "d"), InternalError(NewPath(""), fmt.Errorf("e"))},
		},
		[]int{
			0,
			0,
			1,
			1,
			2,
		},
	}

	if len(testCases.ErrList) != len(testCases.NumExpectedErrs) {
		t.Errorf("Mismatch: length of NumExpectedErrs does not match length of ErrList")
	}
	for i, tc := range testCases.ErrList {
		agg := tc.ToAggregate()
		numErrs := 0

		if agg != nil {
			numErrs = len(agg.Errors())
		}
		if numErrs != testCases.NumExpectedErrs[i] {
			t.Errorf("[%d] Expected %d, got %d", i, testCases.NumExpectedErrs[i], numErrs)
		}

		if len(tc) == 0 {
			if agg != nil {
				t.Errorf("[%d] Expected nil, got %#v", i, agg)
			}
		} else if agg == nil {
			t.Errorf("[%d] Expected non-nil", i)
		}
	}
}

func TestErrListFilter(t *testing.T) {
	list := ErrorList{
		Invalid(NewPath("test.field"), "", ""),
		Invalid(NewPath("field.test"), "", ""),
		Duplicate(NewPath("test"), "value"),
	}
	if len(list.Filter(NewErrorTypeMatcher(ErrorTypeDuplicate))) != 2 {
		t.Errorf("should not filter")
	}
	if len(list.Filter(NewErrorTypeMatcher(ErrorTypeInvalid))) != 1 {
		t.Errorf("should filter")
	}
}

func TestErrorOrigin(t *testing.T) {
	err := Invalid(NewPath("field"), "value", "detail")

	// Test WithOrigin
	newErr := err.WithOrigin("origin1")
	if newErr.Origin != "origin1" {
		t.Errorf("Expected Origin to be 'origin1', got '%s'", newErr.Origin)
	}
	if err.Origin != "origin1" {
		t.Errorf("Expected Origin to be 'origin1', got '%s'", err.Origin)
	}
}

func TestErrorListOrigin(t *testing.T) {
	// Create an ErrorList with multiple errors
	list := ErrorList{
		Invalid(NewPath("field1"), "value1", "detail1"),
		Invalid(NewPath("field2"), "value2", "detail2"),
		Required(NewPath("field3"), "detail3"),
	}

	// Test WithOrigin
	newList := list.WithOrigin("origin1")
	// Check that WithOrigin returns the modified list
	for i, err := range newList {
		if err.Origin != "origin1" {
			t.Errorf("Error %d: Expected Origin to be 'origin2', got '%s'", i, err.Origin)
		}
	}

	// Check that the original list was also modified (WithOrigin modifies and returns the same list)
	for i, err := range list {
		if err.Origin != "origin1" {
			t.Errorf("Error %d: Expected original list Origin to be 'origin2', got '%s'", i, err.Origin)
		}
	}
}

func TestErrorMarkDeclarative(t *testing.T) {
	// Test for single Error
	err := Invalid(NewPath("field"), "value", "detail")
	if err.CoveredByDeclarative {
		t.Errorf("New error should not be declarative by default")
	}

	// Mark as declarative
	err.MarkCoveredByDeclarative() //nolint:errcheck // The "error" here is not an unexpected error from the function.
	if !err.CoveredByDeclarative {
		t.Errorf("Error should be declarative after marking")
	}
}

func TestErrorListMarkDeclarative(t *testing.T) {
	// Test for ErrorList
	list := ErrorList{
		Invalid(NewPath("field1"), "value1", "detail1"),
		Invalid(NewPath("field2"), "value2", "detail2"),
	}

	// Verify none are declarative by default
	for i, err := range list {
		if err.CoveredByDeclarative {
			t.Errorf("Error %d should not be declarative by default", i)
		}
	}

	// Mark list as declarative
	list.MarkCoveredByDeclarative()

	// Verify all errors in the list are now declarative
	for i, err := range list {
		if !err.CoveredByDeclarative {
			t.Errorf("Error %d should be declarative after marking the list", i)
		}
	}
}

func TestErrorListExtractCoveredByDeclarative(t *testing.T) {
	testCases := []struct {
		list         ErrorList
		expectedList ErrorList
	}{
		{
			ErrorList{},
			ErrorList{},
		},
		{
			ErrorList{Invalid(NewPath("field1"), nil, "")},
			ErrorList{},
		},
		{
			ErrorList{Invalid(NewPath("field1"), nil, "").MarkCoveredByDeclarative(), Required(NewPath("field2"), "detail2")},
			ErrorList{Invalid(NewPath("field1"), nil, "").MarkCoveredByDeclarative()},
		},
	}

	for _, tc := range testCases {
		got := tc.list.ExtractCoveredByDeclarative()
		if !reflect.DeepEqual(got, tc.expectedList) {
			t.Errorf("For list %v, expected %v, got %v", tc.list, tc.expectedList, got)
		}
	}
}

func TestErrorListRemoveCoveredByDeclarative(t *testing.T) {
	testCases := []struct {
		list         ErrorList
		expectedList ErrorList
	}{
		{
			ErrorList{},
			ErrorList{},
		},
		{
			ErrorList{Invalid(NewPath("field1"), nil, "").MarkCoveredByDeclarative(), Required(NewPath("field2"), "detail2")},
			ErrorList{Required(NewPath("field2"), "detail2")},
		},
	}

	for _, tc := range testCases {
		got := tc.list.RemoveCoveredByDeclarative()
		if !reflect.DeepEqual(got, tc.expectedList) {
			t.Errorf("For list %v, expected %v, got %v", tc.list, tc.expectedList, got)
		}
	}
}

func TestErrorFormatting(t *testing.T) {
	cases := []struct {
		name   string
		input  *Error
		expect string
	}{{
		name: "required",
		input: &Error{
			Type:                 ErrorTypeRequired,
			Field:                "path.to.field",
			BadValue:             "the value",
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Required value: the details`,
	}, {
		name:   "required func",
		input:  Required(NewPath("path.to.field"), "the details"),
		expect: `path.to.field: Required value: the details`,
	}, {
		name: "forbidden",
		input: &Error{
			Type:                 ErrorTypeForbidden,
			Field:                "path.to.field",
			BadValue:             "the value",
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Forbidden: the details`,
	}, {
		name:   "forbidden func",
		input:  Forbidden(NewPath("path.to.field"), "the details"),
		expect: `path.to.field: Forbidden: the details`,
	}, {
		name: "too long",
		input: &Error{
			Type:                 ErrorTypeTooLong,
			Field:                "path.to.field",
			BadValue:             "the value",
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Too long: the details`,
	}, {
		name:   "too long func(1)",
		input:  TooLong(NewPath("path.to.field"), "the value", 1),
		expect: `path.to.field: Too long: may not be more than 1 byte`,
	}, {
		name:   "too long func(2)",
		input:  TooLong(NewPath("path.to.field"), "the value", 2),
		expect: `path.to.field: Too long: may not be more than 2 bytes`,
	}, {
		name:   "too long func(-1)",
		input:  TooLong(NewPath("path.to.field"), "the value", -1),
		expect: `path.to.field: Too long: value is too long`,
	}, {
		name: "too many",
		input: &Error{
			Type:                 ErrorTypeTooMany,
			Field:                "path.to.field",
			BadValue:             "the value",
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Too many: "the value": the details`,
	}, {
		name:   "too many func(2, 1)",
		input:  TooMany(NewPath("path.to.field"), 2, 1),
		expect: `path.to.field: Too many: 2: must have at most 1 item`,
	}, {
		name:   "too many func(3, 2)",
		input:  TooMany(NewPath("path.to.field"), 3, 2),
		expect: `path.to.field: Too many: 3: must have at most 2 items`,
	}, {
		name:   "too many func(2, -1)",
		input:  TooMany(NewPath("path.to.field"), 2, -1),
		expect: `path.to.field: Too many: 2: has too many items`,
	}, {
		name:   "too many func(-1, 1)",
		input:  TooMany(NewPath("path.to.field"), -1, 1),
		expect: `path.to.field: Too many: must have at most 1 item`,
	}, {
		name: "internal error",
		input: &Error{
			Type:                 ErrorTypeInternal,
			Field:                "path.to.field",
			BadValue:             "the value",
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Internal error: the details`,
	}, {
		name:   "internal error func",
		input:  InternalError(NewPath("path.to.field"), fmt.Errorf("the error")),
		expect: `path.to.field: Internal error: the error`,
	}, {
		name: "invalid string",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             "the value",
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: "the value": the details`,
	}, {
		name: "invalid string type",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             StringType("the value"),
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: "the value": the details`,
	}, {
		name: "invalid int",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             -42,
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: -42: the details`,
	}, {
		name: "invalid bool",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             true,
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: true: the details`,
	}, {
		name: "invalid struct",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             mkTinyStruct(),
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: {"stringField":"stringval","intField":9376,"boolField":true}: the details`,
	}, {
		name: "invalid list",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             []string{"one", "two", "three"},
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: ["one","two","three"]: the details`,
	}, {
		name: "invalid map",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             map[string]int{"one": 1, "two": 2, "three": 3},
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: {"one":1,"three":3,"two":2}: the details`,
	}, {
		name: "invalid time",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             time.Time{},
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: "0001-01-01T00:00:00Z": the details`,
	}, {
		name: "invalid omitValue",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             omitValue,
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: the details`,
	}, {
		name: "invalid untyped nil",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             nil,
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: null: the details`,
	}, {
		name: "invalid typed nil",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             (*string)(nil),
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: null: the details`,
	}, {
		name: "invalid string ptr",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             ptr.To("the value"),
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: "the value": the details`,
	}, {
		name: "invalid string type ptr",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             ptr.To(StringType("the value")),
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: "the value": the details`,
	}, {
		name: "invalid int ptr",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             ptr.To(-42),
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: -42: the details`,
	}, {
		name: "invalid bool ptr",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             ptr.To(true),
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: true: the details`,
	}, {
		name: "invalid struct ptr",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             ptr.To(mkTinyStruct()),
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: {"stringField":"stringval","intField":9376,"boolField":true}: the details`,
	}, {
		name: "invalid list ptr",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             ptr.To([]string{"one", "two", "three"}),
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: ["one","two","three"]: the details`,
	}, {
		name: "invalid map ptr",
		input: &Error{
			Type:                 ErrorTypeInvalid,
			Field:                "path.to.field",
			BadValue:             ptr.To(map[string]int{"one": 1, "two": 2, "three": 3}),
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: {"one":1,"three":3,"two":2}: the details`,
	}, {
		name:   "invalid func",
		input:  Invalid(NewPath("path.to.field"), "the value", "the details"),
		expect: `path.to.field: Invalid value: "the value": the details`,
	}, {
		name: "not found",
		input: &Error{
			Type:                 ErrorTypeNotFound,
			Field:                "path.to.field",
			BadValue:             "the value",
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Not found: "the value": the details`,
	}, {
		name: "not supported",
		input: &Error{
			Type:                 ErrorTypeNotSupported,
			Field:                "path.to.field",
			BadValue:             "the value",
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Unsupported value: "the value": the details`,
	}, {
		name:   "not supported func",
		input:  NotSupported(NewPath("path.to.field"), "the value", []string{"val1", "val2"}),
		expect: `path.to.field: Unsupported value: "the value": supported values: "val1", "val2"`,
	}, {
		name: "duplicate",
		input: &Error{
			Type:                 ErrorTypeDuplicate,
			Field:                "path.to.field",
			BadValue:             "the value",
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Duplicate value: "the value": the details`,
	}, {
		name:   "duplicate func",
		input:  Duplicate(NewPath("path.to.field"), "the value"),
		expect: `path.to.field: Duplicate value: "the value"`,
	}, {
		name: "type invalid",
		input: &Error{
			Type:                 ErrorTypeTypeInvalid,
			Field:                "path.to.field",
			BadValue:             "the value",
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: "the value": the details`,
	}, {
		name:   "type invalid func",
		input:  TypeInvalid(NewPath("path.to.field"), "the value", "the details"),
		expect: `path.to.field: Invalid value: "the value": the details`,
	}, {
		name: "failed marshal stringer",
		input: &Error{
			Type:                 ErrorTypeTypeInvalid,
			Field:                "path.to.field",
			BadValue:             SelfMarshalerStringer{"invisible"},
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: magic: the details`,
	}, {
		name: "failed marshal non-stringer",
		input: &Error{
			Type:                 ErrorTypeTypeInvalid,
			Field:                "path.to.field",
			BadValue:             SelfMarshalerNonStringer{"visible"},
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Invalid value: field.SelfMarshalerNonStringer{S:"visible"}: the details`,
	}, {
		name: "unknown error type",
		input: &Error{
			Type:                 "not real",
			Field:                "path.to.field",
			BadValue:             SelfMarshalerNonStringer{"visible"},
			Detail:               "the details",
			Origin:               "theOrigin",
			CoveredByDeclarative: true,
		},
		expect: `path.to.field: Internal error: unhandled error code: <unknown error "not real">: please report this: the details`,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := tc.input.Error()
			if want := tc.expect; result != want {
				t.Errorf("wrong error string:\n  expected: %q\n       got: %q", want, result)
			}
		})
	}
}

type StringType string

type TinyStruct struct {
	StringField string `json:"stringField"`
	IntField    int    `json:"intField"`
	BoolField   bool   `json:"boolField"`
}

func mkTinyStruct() TinyStruct {
	return TinyStruct{
		StringField: "stringval",
		IntField:    9376,
		BoolField:   true,
	}
}

type SelfMarshalerStringer struct{ S string }

func (SelfMarshalerStringer) MarshalJSON() ([]byte, error) {
	return nil, fmt.Errorf("this always fails")
}

func (SelfMarshalerStringer) String() string {
	return "magic"
}

type SelfMarshalerNonStringer struct{ S string }

func (SelfMarshalerNonStringer) MarshalJSON() ([]byte, error) {
	return nil, fmt.Errorf("this always fails")
}
