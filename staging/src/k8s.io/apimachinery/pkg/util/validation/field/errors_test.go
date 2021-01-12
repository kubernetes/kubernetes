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
	"errors"
	"fmt"
	"strings"
	"testing"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
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
			func() *Error { return NotSupported(NewPath("f"), "v", nil) },
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

func TestErrorUsefulMessage(t *testing.T) {
	{
		s := Invalid(nil, nil, "").Error()
		t.Logf("message: %v", s)
		if !strings.Contains(s, "null") {
			t.Errorf("error message did not contain 'null': %s", s)
		}
	}

	s := Invalid(NewPath("foo"), "bar", "deet").Error()
	t.Logf("message: %v", s)
	for _, part := range []string{"foo", "bar", "deet", ErrorTypeInvalid.String()} {
		if !strings.Contains(s, part) {
			t.Errorf("error message did not contain expected part '%v'", part)
		}
	}

	type complicated struct {
		Baz   int
		Qux   string
		Inner interface{}
		KV    map[string]int
	}
	s = Invalid(
		NewPath("foo"),
		&complicated{
			Baz:   1,
			Qux:   "aoeu",
			Inner: &complicated{Qux: "asdf"},
			KV:    map[string]int{"Billy": 2},
		},
		"detail",
	).Error()
	t.Logf("message: %v", s)
	for _, part := range []string{
		"foo", ErrorTypeInvalid.String(),
		"Baz", "Qux", "Inner", "KV", "detail",
		"1", "aoeu", "Billy", "2",
		// "asdf", TODO: re-enable once we have a better nested printer
	} {
		if !strings.Contains(s, part) {
			t.Errorf("error message did not contain expected part '%v'", part)
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

func TestNotSupported(t *testing.T) {
	notSupported := NotSupported(NewPath("f"), "v", []string{"a", "b", "c"})
	expected := `Unsupported value: "v": supported values: "a", "b", "c"`
	if notSupported.ErrorBody() != expected {
		t.Errorf("Expected: %s\n, but got: %s\n", expected, notSupported.ErrorBody())
	}
}

func TestErrorIs(t *testing.T) {

	testCases := []struct {
		name      string
		targetErr *Error
		cmpErr    error
		wantIs    bool
	}{
		{
			"Valid test with Invalid error",
			Invalid(NewPath("f"), "v", "d"),
			&Error{ErrorTypeInvalid, NewPath("f").String(), "v", "d"},
			true,
		},
		{
			"Valid test with Invalid error using wrapping",
			Invalid(NewPath("f"), "v", "d"),
			fmt.Errorf("%w", Invalid(NewPath("f"), "v", "d")),
			true,
		},
		{
			"Invalid test with Invalid error",
			Invalid(NewPath("f"), "v", "d"),
			&Error{ErrorTypeInvalid, NewPath("f").String(), "dv", "d"},
			false,
		},
		{
			"Valid test with NotSupported error",
			NotSupported(NewPath("f"), "v", []string{"a", "b", "c"}),
			&Error{ErrorTypeNotSupported, NewPath("f").String(), "v", `supported values: "a", "b", "c"`},
			true,
		},
		{
			"Invalid test with NotSupported error",
			NotSupported(NewPath("f"), "v", []string{"a", "b", "c"}),
			&Error{ErrorTypeNotSupported, NewPath("f").String(), "dv", `supported values: "a", "b", "c"`},
			false,
		},
		{
			"Valid test with Forbidden error",
			Forbidden(NewPath("f"), "d"),
			&Error{ErrorTypeForbidden, NewPath("f").String(), "", "d"},
			true,
		},
		{
			"Invalid test with Forbidden error",
			Forbidden(NewPath("f"), "d"),
			&Error{ErrorTypeForbidden, NewPath("f").String(), "dv", `supported values: "a", "b", "c"`},
			false,
		},
		{
			"Valid test with TooLong error",
			TooLong(NewPath("f"), "v", 500),
			&Error{ErrorTypeTooLong, NewPath("f").String(), "v", "must have at most 500 bytes"},
			true,
		},
		{
			"Invalid test with TooLong error",
			TooLong(NewPath("f"), "v", 500),
			&Error{ErrorTypeTooLong, NewPath("f").String(), "v", "must have at most 5001 bytes"},
			false,
		},
		{
			"Valid test with TooMany error",
			TooMany(NewPath("f"), 500, 100),
			&Error{ErrorTypeTooMany, NewPath("f").String(), 500, "must have at most 100 items"},
			true,
		},
		{
			"Invalid test with TooMany error",
			TooMany(NewPath("f"), 500, 100),
			&Error{ErrorTypeTooMany, NewPath("f").String(), "v", "must have at most 5001 items"},
			false,
		},
		{
			"Valid test with Duplicate error",
			Duplicate(NewPath("f"), "v"),
			&Error{ErrorTypeDuplicate, NewPath("f").String(), "v", ""},
			true,
		},
		{
			"Invalid test with Duplicate error",
			Duplicate(NewPath("f"), "v"),
			&Error{ErrorTypeDuplicate, NewPath("df").String(), "v", ""},
			false,
		},
		{
			"Valid test with NotFound error",
			NotFound(NewPath("f"), "v"),
			&Error{ErrorTypeNotFound, NewPath("f").String(), "v", ""},
			true,
		},
		{
			"Invalid test with NotFound error",
			NotFound(NewPath("f"), "v"),
			&Error{ErrorTypeNotFound, NewPath("f").String(), []string{"a", "b", "c"}, "d"},
			false,
		},
		{
			"Valid test with Required error",
			Required(NewPath("f"), "d"),
			&Error{ErrorTypeRequired, NewPath("f").String(), "", "d"},
			true,
		},
		{
			"Invalid test with Required error",
			Required(NewPath("f"), "d"),
			&Error{ErrorTypeNotFound, NewPath("f").String(), "", "d"},
			false,
		},
		{
			"Valid test with InternalError error",
			InternalError(NewPath("f"), fmt.Errorf("e")),
			&Error{ErrorTypeInternal, NewPath("f").String(), nil, fmt.Errorf("e").Error()},
			true,
		},
		{
			"Invalid test with InternalError error",
			InternalError(NewPath("f"), fmt.Errorf("e")),
			&Error{ErrorTypeInternal, NewPath("f").String(), "v", fmt.Errorf("e").Error()},
			false,
		},
		{
			"Valid test with InternalError using wrapping",
			InternalError(NewPath("f"), fmt.Errorf("ex")),
			fmt.Errorf("%w", InternalError(NewPath("f"), fmt.Errorf("ex"))),
			true,
		},
		{
			"Valid test with Aggregate errors",
			Invalid(NewPath("f"), "v", "d"),
			utilerrors.NewAggregate([]error{Invalid(NewPath("f"), "v", "d"), Invalid(NewPath("a"), "b", "c")}),
			true,
		},
		{
			"Invalid test with Aggregate errors",
			Invalid(NewPath("f"), "v", "d"),
			utilerrors.NewAggregate([]error{Invalid(NewPath("diff"), "v", "d"), Invalid(NewPath("a"), "b", "c")}),
			false,
		},
	}

	for _, testCase := range testCases {
		if errors.Is(testCase.cmpErr, testCase.targetErr) != testCase.wantIs {
			t.Errorf("Test case: %s. cmpError: %q , targetErr: %q", testCase.name, testCase.cmpErr, testCase.targetErr)
		}
	}
}
