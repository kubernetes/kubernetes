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

package fielderrors

import (
	"strings"
	"testing"
)

func TestMakeFuncs(t *testing.T) {
	testCases := []struct {
		fn       func() *ValidationError
		expected ValidationErrorType
	}{
		{
			func() *ValidationError { return NewFieldInvalid("f", "v", "d") },
			ValidationErrorTypeInvalid,
		},
		{
			func() *ValidationError { return NewFieldValueNotSupported("f", "v", nil) },
			ValidationErrorTypeNotSupported,
		},
		{
			func() *ValidationError { return NewFieldDuplicate("f", "v") },
			ValidationErrorTypeDuplicate,
		},
		{
			func() *ValidationError { return NewFieldNotFound("f", "v") },
			ValidationErrorTypeNotFound,
		},
		{
			func() *ValidationError { return NewFieldRequired("f") },
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

func TestValidationErrorUsefulMessage(t *testing.T) {
	s := NewFieldInvalid("foo", "bar", "deet").Error()
	t.Logf("message: %v", s)
	for _, part := range []string{"foo", "bar", "deet", ValidationErrorTypeInvalid.String()} {
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
	s = NewFieldInvalid(
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
		"foo", ValidationErrorTypeInvalid.String(),
		"Baz", "Qux", "Inner", "KV", "detail",
		"1", "aoeu", "asdf", "Billy", "2",
	} {
		if !strings.Contains(s, part) {
			t.Errorf("error message did not contain expected part '%v'", part)
		}
	}
}

func TestErrListFilter(t *testing.T) {
	list := ValidationErrorList{
		NewFieldInvalid("test.field", "", ""),
		NewFieldInvalid("field.test", "", ""),
		NewFieldDuplicate("test", "value"),
	}
	if len(list.Filter(NewValidationErrorTypeMatcher(ValidationErrorTypeDuplicate))) != 2 {
		t.Errorf("should not filter")
	}
	if len(list.Filter(NewValidationErrorTypeMatcher(ValidationErrorTypeInvalid))) != 1 {
		t.Errorf("should filter")
	}
	if len(list.Filter(NewValidationErrorFieldPrefixMatcher("test"))) != 1 {
		t.Errorf("should filter")
	}
	if len(list.Filter(NewValidationErrorFieldPrefixMatcher("test."))) != 2 {
		t.Errorf("should filter")
	}
	if len(list.Filter(NewValidationErrorFieldPrefixMatcher(""))) != 0 {
		t.Errorf("should filter")
	}
	if len(list.Filter(NewValidationErrorFieldPrefixMatcher("field."), NewValidationErrorTypeMatcher(ValidationErrorTypeDuplicate))) != 1 {
		t.Errorf("should filter")
	}
}

func TestErrListPrefix(t *testing.T) {
	testCases := []struct {
		Err      *ValidationError
		Expected string
	}{
		{
			NewFieldNotFound("[0].bar", "value"),
			"foo[0].bar",
		},
		{
			NewFieldInvalid("field", "value", ""),
			"foo.field",
		},
		{
			NewFieldDuplicate("", "value"),
			"foo",
		},
	}
	for _, testCase := range testCases {
		errList := ValidationErrorList{testCase.Err}
		prefix := errList.Prefix("foo")
		if prefix == nil || len(prefix) != len(errList) {
			t.Errorf("Prefix should return self")
		}
		if e, a := testCase.Expected, errList[0].(*ValidationError).Field; e != a {
			t.Errorf("expected %s, got %s", e, a)
		}
	}
}

func TestErrListPrefixIndex(t *testing.T) {
	testCases := []struct {
		Err      *ValidationError
		Expected string
	}{
		{
			NewFieldNotFound("[0].bar", "value"),
			"[1][0].bar",
		},
		{
			NewFieldInvalid("field", "value", ""),
			"[1].field",
		},
		{
			NewFieldDuplicate("", "value"),
			"[1]",
		},
	}
	for _, testCase := range testCases {
		errList := ValidationErrorList{testCase.Err}
		prefix := errList.PrefixIndex(1)
		if prefix == nil || len(prefix) != len(errList) {
			t.Errorf("PrefixIndex should return self")
		}
		if e, a := testCase.Expected, errList[0].(*ValidationError).Field; e != a {
			t.Errorf("expected %s, got %s", e, a)
		}
	}
}
