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

package errors

import (
	"flag"
	"strings"
	"testing"
)

func TestCallStack(t *testing.T) {
	testLeafFunc(t)
}

func testLeafFunc(t *testing.T) {
	e := New("foo")
	s := StackTrace(e)
	if !strings.HasPrefix(s, "k8s.io/kubernetes/cmd/kubeadm/app/util/errors.testLeafFunc") {
		t.Fatalf("unexpected stack trace")
	}
}

func TestHandleError(t *testing.T) {
	_ = flag.CommandLine.String("v", "", "")

	tests := []struct {
		name             string
		vValue           string
		errMessage       string
		expectedContains []string
	}{
		{
			name:             "without stack trace",
			vValue:           "1",
			errMessage:       "foo",
			expectedContains: []string{"foo"},
		},
		{
			name:             "with stack trace",
			vValue:           "5",
			errMessage:       "bar",
			expectedContains: []string{"bar", "errors.TestHandleError"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_ = flag.CommandLine.Set("v", tc.vValue)
			flag.Parse()

			err := New(tc.errMessage)

			handleFunc := func(msg string, code int) {
				if code != defaultErrorExitCode {
					t.Errorf("expected error code: %d", defaultErrorExitCode)
				}
				for _, c := range tc.expectedContains {
					if !strings.Contains(msg, c) {
						t.Errorf("expected error message to contain: %s", c)
					}
				}
			}

			handleError(err, handleFunc)
		})
	}
}

// customError is used to check that As can find a wrapped error by type.
type customError struct {
	msg string
}

func (e *customError) Error() string {
	return e.msg
}

func TestWrappedErrors(t *testing.T) {
	cause := &customError{msg: "cause"}
	inner := WithMessage(cause, "inner")

	tests := []struct {
		name            string
		err             error
		expectedMessage string
		expectedUnwrap  error
	}{
		{
			name:            "Wrap",
			err:             Wrap(cause, "context"),
			expectedMessage: "context: cause",
			expectedUnwrap:  cause,
		},
		{
			name:            "Wrapf",
			err:             Wrapf(cause, "context %d", 1),
			expectedMessage: "context 1: cause",
			expectedUnwrap:  cause,
		},
		{
			name:            "WithMessage",
			err:             WithMessage(cause, "context"),
			expectedMessage: "context: cause",
			expectedUnwrap:  cause,
		},
		{
			name:            "WithMessagef",
			err:             WithMessagef(cause, "context %d", 1),
			expectedMessage: "context 1: cause",
			expectedUnwrap:  cause,
		},
		{
			name:            "Errorf with %w",
			err:             Errorf("context %d: %w", 1, cause),
			expectedMessage: "context 1: cause",
			expectedUnwrap:  cause,
		},
		{
			name:            "Wrap of an already wrapped error",
			err:             Wrap(inner, "outer"),
			expectedMessage: "outer: inner: cause",
			expectedUnwrap:  inner,
		},
		{
			name:            "Errorf without %w",
			err:             Errorf("context %d", 1),
			expectedMessage: "context 1",
		},
		{
			name:            "New",
			err:             New("context"),
			expectedMessage: "context",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.err.Error() != tc.expectedMessage {
				t.Errorf("expected message: %q, got: %q", tc.expectedMessage, tc.err.Error())
			}
			if u := Unwrap(tc.err); u != tc.expectedUnwrap {
				t.Errorf("expected Unwrap: %v, got: %v", tc.expectedUnwrap, u)
			}

			// An error that wraps nothing must not report the cause as found.
			expectedFound := tc.expectedUnwrap != nil
			if found := Is(tc.err, cause); found != expectedFound {
				t.Errorf("expected Is: %v, got: %v", expectedFound, found)
			}
			var target *customError
			if found := As(tc.err, &target); found != expectedFound {
				t.Errorf("expected As: %v, got: %v", expectedFound, found)
			}

			if s := StackTrace(tc.err); s == "no stack trace" {
				t.Errorf("expected a stack trace")
			}
		})
	}
}

func TestWrapNilError(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{
			name: "Wrap",
			err:  Wrap(nil, "context"),
		},
		{
			name: "Wrapf",
			err:  Wrapf(nil, "context %d", 1),
		},
		{
			name: "WithMessage",
			err:  WithMessage(nil, "context"),
		},
		{
			name: "WithMessagef",
			err:  WithMessagef(nil, "context %d", 1),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.err != nil {
				t.Errorf("expected nil error, got: %v", tc.err)
			}
		})
	}
}

func TestErrorfWithMultipleWrappedErrors(t *testing.T) {
	first := &customError{msg: "first"}
	second := &customError{msg: "second"}
	err := Errorf("context: %w and %w", first, second)

	expectedMessage := "context: first and second"
	if err.Error() != expectedMessage {
		t.Errorf("expected message: %q, got: %q", expectedMessage, err.Error())
	}
	if !Is(err, first) {
		t.Errorf("expected Is to find the first wrapped error")
	}
	if !Is(err, second) {
		t.Errorf("expected Is to find the second wrapped error")
	}
	var target *customError
	if !As(err, &target) {
		t.Errorf("expected As to find a wrapped error")
	}
}
