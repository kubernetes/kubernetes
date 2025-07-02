// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package testutil provides utility functions for testing in Go.
package testutil

import (
	"cmp"
	"errors"
	"reflect"
	"testing"
)

var ErrSample = errors.New("an error")

type T interface {
	Helper()
	Errorf(format string, args ...interface{})
	Fatalf(format string, args ...interface{})
}

func assert(t T, condition bool, msgAndArgs ...interface{}) bool {
	t.Helper()
	//nolint:nestif // Acceptable for readability
	if !condition {
		if len(msgAndArgs) > 0 {
			if format, ok := msgAndArgs[0].(string); ok && len(msgAndArgs) > 1 {
				t.Errorf(format, msgAndArgs[1:]...)
			} else {
				t.Errorf("%v", msgAndArgs...)
			}
		} else {
			t.Errorf("assert failed")
		}
		return false
	}
	return true
}

func require(t *testing.T, condition bool, msgAndArgs ...interface{}) {
	t.Helper()
	if !condition {
		t.Fatalf("require failed: %s", msgAndArgs)
	}
}

func AssertEqual[T cmp.Ordered](t *testing.T, expected, actual T, msgAndArgs ...interface{}) bool {
	t.Helper()
	got := cmp.Compare(expected, actual)
	return assert(
		t,
		got == 0,
		"assertEqual failed: expected %q, got %q, %s",
		expected,
		actual,
		msgAndArgs,
	)
}

func AssertTrue(t *testing.T, condition bool, msgAndArgs ...interface{}) bool {
	t.Helper()
	return assert(t, condition, "assertTrue failed: condition is false, %s", msgAndArgs)
}

func hasError(t *testing.T, err error) bool {
	t.Helper()
	return err != nil
}

func RequireError(t *testing.T, err error, msgAndArgs ...interface{}) {
	t.Helper()
	got := hasError(t, err)
	require(t, got, "requireError failed: expected error, got nil, %s", msgAndArgs)
}

func RequireNoError(t *testing.T, err error, msgAndArgs ...interface{}) {
	t.Helper()
	got := hasError(t, err)
	require(t, !got, "requireNoError failed: expected no error, got %v, %s", err, msgAndArgs)
}

func RequireErrorIs(t *testing.T, err error, target error, msgAndArgs ...interface{}) {
	t.Helper()
	got := errors.Is(err, target)
	require(t, got, "requireErrorIs failed: expected error %v, got %v, %s", target, err, msgAndArgs)
}

func RequireErrorAs(t *testing.T, err error, target interface{}, msgAndArgs ...interface{}) {
	t.Helper()
	got := errors.As(err, target)
	require(
		t,
		got,
		"requireErrorAs failed: expected error to be of type %T, got %v, %s",
		target,
		err,
		msgAndArgs,
	)
}

func RequireTrue(t *testing.T, condition bool, msgAndArgs ...interface{}) {
	t.Helper()
	require(
		t,
		condition,
		"requireTrue failed: expected condition to be true, got false, %s",
		msgAndArgs,
	)
}

// From github.com/stretchr/testify/assert
// Copyright (c) 2012-2020 Mat Ryer, Tyler Bunnell and contributors.
// Licensed under the MIT License (MIT).
func isNil(object interface{}) bool {
	if object == nil {
		return true
	}

	value := reflect.ValueOf(object)
	switch value.Kind() { //nolint:exhaustive // exhaustive is not needed here, as we handle all common cases
	case
		reflect.Chan, reflect.Func,
		reflect.Interface, reflect.Map,
		reflect.Ptr, reflect.Slice, reflect.UnsafePointer:

		return value.IsNil()
	}

	return false
}

func AssertNil(t *testing.T, object interface{}, msgAndArgs ...interface{}) bool {
	t.Helper()
	got := isNil(object)
	return assert(
		t,
		got,
		"assertNil failed: expected nil, got %v, %s",
		object,
		msgAndArgs,
	)
}

func AssertNotNil(t *testing.T, object interface{}, msgAndArgs ...interface{}) bool {
	t.Helper()
	got := !isNil(object)
	return assert(
		t,
		got,
		"assertNotNil failed: expected not nil, got nil, %s",
		msgAndArgs,
	)
}

func RequireNotNil(t *testing.T, object interface{}, msgAndArgs ...interface{}) {
	t.Helper()
	got := isNil(object)
	require(
		t,
		!got,
		"requireNotNil failed: expected not nil, got nil, %s",
		msgAndArgs,
	)
}

func RequirePanics(t *testing.T, f func(), msgAndArgs ...interface{}) bool {
	t.Helper()
	defer func() {
		got := recover()
		require(
			t,
			got != nil,
			"requirePanics failed: expected panic, got none, %s",
			msgAndArgs,
		)
	}()
	f()
	return true
}
