/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package validation

import (
	"fmt"
	"strings"
	"testing"
)

func TestMakeFuncs(t *testing.T) {
	testCases := []struct {
		fn       func() *Error
		expected ErrorType
	}{
		{
			func() *Error { return NewInvalidError("f", "v", "d") },
			ErrorTypeInvalid,
		},
		{
			func() *Error { return NewNotSupportedError("f", "v", nil) },
			ErrorTypeNotSupported,
		},
		{
			func() *Error { return NewDuplicateError("f", "v") },
			ErrorTypeDuplicate,
		},
		{
			func() *Error { return NewNotFoundError("f", "v") },
			ErrorTypeNotFound,
		},
		{
			func() *Error { return NewRequiredError("f") },
			ErrorTypeRequired,
		},
		{
			func() *Error { return NewInternalError("f", fmt.Errorf("e")) },
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
	s := NewInvalidError("foo", "bar", "deet").Error()
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
	s = NewInvalidError(
		"foo",
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
		"1", "aoeu", "asdf", "Billy", "2",
	} {
		if !strings.Contains(s, part) {
			t.Errorf("error message did not contain expected part '%v'", part)
		}
	}
}

func TestToAggregate(t *testing.T) {
	testCases := []ErrorList{
		nil,
		{},
		{NewInvalidError("f", "v", "d")},
		{NewInvalidError("f", "v", "d"), NewInternalError("", fmt.Errorf("e"))},
	}
	for i, tc := range testCases {
		agg := tc.ToAggregate()
		if len(tc) == 0 {
			if agg != nil {
				t.Errorf("[%d] Expected nil, got %#v", i, agg)
			}
		} else if agg == nil {
			t.Errorf("[%d] Expected non-nil", i)
		} else if len(tc) != len(agg.Errors()) {
			t.Errorf("[%d] Expected %d, got %d", i, len(tc), len(agg.Errors()))
		}
	}
}

func TestErrListFilter(t *testing.T) {
	list := ErrorList{
		NewInvalidError("test.field", "", ""),
		NewInvalidError("field.test", "", ""),
		NewDuplicateError("test", "value"),
	}
	if len(list.Filter(NewErrorTypeMatcher(ErrorTypeDuplicate))) != 2 {
		t.Errorf("should not filter")
	}
	if len(list.Filter(NewErrorTypeMatcher(ErrorTypeInvalid))) != 1 {
		t.Errorf("should filter")
	}
}

func TestErrListPrefix(t *testing.T) {
	testCases := []struct {
		Err      *Error
		Expected string
	}{
		{
			NewNotFoundError("[0].bar", "value"),
			"foo[0].bar",
		},
		{
			NewInvalidError("field", "value", ""),
			"foo.field",
		},
		{
			NewDuplicateError("", "value"),
			"foo",
		},
	}
	for _, testCase := range testCases {
		errList := ErrorList{testCase.Err}
		prefix := errList.Prefix("foo")
		if prefix == nil || len(prefix) != len(errList) {
			t.Errorf("Prefix should return self")
		}
		if e, a := testCase.Expected, errList[0].Field; e != a {
			t.Errorf("expected %s, got %s", e, a)
		}
	}
}

func TestErrListPrefixIndex(t *testing.T) {
	testCases := []struct {
		Err      *Error
		Expected string
	}{
		{
			NewNotFoundError("[0].bar", "value"),
			"[1][0].bar",
		},
		{
			NewInvalidError("field", "value", ""),
			"[1].field",
		},
		{
			NewDuplicateError("", "value"),
			"[1]",
		},
	}
	for _, testCase := range testCases {
		errList := ErrorList{testCase.Err}
		prefix := errList.PrefixIndex(1)
		if prefix == nil || len(prefix) != len(errList) {
			t.Errorf("PrefixIndex should return self")
		}
		if e, a := testCase.Expected, errList[0].Field; e != a {
			t.Errorf("expected %s, got %s", e, a)
		}
	}
}
