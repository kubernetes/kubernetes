/*
Copyright 2014 Google Inc. All rights reserved.

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

package errors

import (
	"fmt"
	"strings"
	"testing"
)

func TestMakeFuncs(t *testing.T) {
	testCases := []struct {
		fn       func() ValidationError
		expected ValidationErrorType
	}{
		{
			func() ValidationError { return NewInvalid("f", "v") },
			ValidationErrorTypeInvalid,
		},
		{
			func() ValidationError { return NewNotSupported("f", "v") },
			ValidationErrorTypeNotSupported,
		},
		{
			func() ValidationError { return NewDuplicate("f", "v") },
			ValidationErrorTypeDuplicate,
		},
		{
			func() ValidationError { return NewNotFound("f", "v") },
			ValidationErrorTypeNotFound,
		},
		{
			func() ValidationError { return NewRequired("f", "v") },
			ValidationErrorTypeRequired,
		},
	}

	for _, testCase := range testCases {
		err := testCase.fn()
		if err.Type != testCase.expected {
			t.Errorf("expected Type %q, got %q", testCase.expected, err.Type)
		}
	}
}

func TestValidationError(t *testing.T) {
	s := NewInvalid("foo", "bar").Error()
	if !strings.Contains(s, "foo") || !strings.Contains(s, "bar") || !strings.Contains(s, ValueOf(ValidationErrorTypeInvalid)) {
		t.Errorf("error message did not contain expected values, got %s", s)
	}
}

func TestErrorList(t *testing.T) {
	errList := ErrorList{}
	if a := errList.ToError(); a != nil {
		t.Errorf("unexpected non-nil error for empty list: %v", a)
	}
	if a := errorListInternal(errList).Error(); a != "" {
		t.Errorf("expected empty string, got %v", a)
	}
	errList = append(errList, NewInvalid("field", "value"))
	// The fact that this compiles is the test.
}

func TestErrorListToError(t *testing.T) {
	errList := ErrorList{}
	err := errList.ToError()
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}

	testCases := []struct {
		errs     ErrorList
		expected string
	}{
		{ErrorList{fmt.Errorf("abc")}, "abc"},
		{ErrorList{fmt.Errorf("abc"), fmt.Errorf("123")}, "abc; 123"},
	}
	for _, testCase := range testCases {
		err := testCase.errs.ToError()
		if err == nil {
			t.Errorf("expected an error, got nil: ErrorList=%v", testCase)
			continue
		}
		if err.Error() != testCase.expected {
			t.Errorf("expected %q, got %q", testCase.expected, err.Error())
		}
	}
}

func TestErrListPrefix(t *testing.T) {
	testCases := []struct {
		Err      ValidationError
		Expected string
	}{
		{
			NewNotFound("[0].bar", "value"),
			"foo[0].bar",
		},
		{
			NewInvalid("field", "value"),
			"foo.field",
		},
		{
			NewDuplicate("", "value"),
			"foo",
		},
	}
	for _, testCase := range testCases {
		errList := ErrorList{testCase.Err}
		prefix := errList.Prefix("foo")
		if prefix == nil || len(prefix) != len(errList) {
			t.Errorf("Prefix should return self")
		}
		if e, a := testCase.Expected, errList[0].(ValidationError).Field; e != a {
			t.Errorf("expected %s, got %s", e, a)
		}
	}
}

func TestErrListPrefixIndex(t *testing.T) {
	testCases := []struct {
		Err      ValidationError
		Expected string
	}{
		{
			NewNotFound("[0].bar", "value"),
			"[1][0].bar",
		},
		{
			NewInvalid("field", "value"),
			"[1].field",
		},
		{
			NewDuplicate("", "value"),
			"[1]",
		},
	}
	for _, testCase := range testCases {
		errList := ErrorList{testCase.Err}
		prefix := errList.PrefixIndex(1)
		if prefix == nil || len(prefix) != len(errList) {
			t.Errorf("PrefixIndex should return self")
		}
		if e, a := testCase.Expected, errList[0].(ValidationError).Field; e != a {
			t.Errorf("expected %s, got %s", e, a)
		}
	}
}
