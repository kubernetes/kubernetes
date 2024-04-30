/*
Copyright 2024 The Kubernetes Authors.

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

package ktesting

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/format"
)

// FailureError is an error where the error string is meant to be passed to
// [TContext.Fatal] directly, i.e. adding some prefix like "unexpected error" is not
// necessary. It is also not necessary to dump the error struct.
type FailureError struct {
	Msg            string
	FullStackTrace string
}

func (f FailureError) Error() string {
	return f.Msg
}

func (f FailureError) Backtrace() string {
	return f.FullStackTrace
}

func (f FailureError) Is(target error) bool {
	return target == ErrFailure
}

// ErrFailure is an empty error that can be wrapped to indicate that an error
// is a FailureError. It can also be used to test for a FailureError:.
//
//	return fmt.Errorf("some problem%w", ErrFailure)
//	...
//	err := someOperation()
//	if errors.Is(err, ErrFailure) {
//	    ...
//	}
var ErrFailure error = FailureError{}

func expect(tCtx TContext, actual interface{}, extra ...interface{}) gomega.Assertion {
	tCtx.Helper()
	return gomega.NewWithT(tCtx).Expect(actual, extra...)
}

func expectNoError(tCtx TContext, err error, explain ...interface{}) {
	if err == nil {
		return
	}

	tCtx.Helper()

	description := buildDescription(explain...)

	if errors.Is(err, ErrFailure) {
		var failure FailureError
		if errors.As(err, &failure) {
			if backtrace := failure.Backtrace(); backtrace != "" {
				if description != "" {
					tCtx.Log(description)
				}
				tCtx.Logf("Failed at:\n    %s", strings.ReplaceAll(backtrace, "\n", "\n    "))
			}
		}
		if description != "" {
			tCtx.Fatalf("%s: %s", description, err.Error())
		}
		tCtx.Fatal(err.Error())
	}

	if description == "" {
		description = "Unexpected error"
	}
	tCtx.Logf("%s:\n%s", description, format.Object(err, 1))
	tCtx.Fatalf("%s: %v", description, err.Error())
}

func buildDescription(explain ...interface{}) string {
	switch len(explain) {
	case 0:
		return ""
	case 1:
		if describe, ok := explain[0].(func() string); ok {
			return describe()
		}
	}
	return fmt.Sprintf(explain[0].(string), explain[1:]...)
}

// Eventually wraps [gomega.Eventually] such that a failure will be reported via
// TContext.Fatal.
//
// In contrast to [gomega.Eventually], the parameter is strongly typed. It must
// accept a TContext as first argument and return one value, the one which is
// then checked with the matcher.
//
// In contrast to direct usage of [gomega.Eventually], make additional
// assertions inside the callback is okay as long as they use the TContext that
// is passed in. For example, errors can be checked with ExpectNoError:
//
//	cb := func(func(tCtx ktesting.TContext) int {
//	    value, err := doSomething(...)
//	    tCtx.ExpectNoError(err, "something failed")
//	    assert(tCtx, 42, value, "the answer")
//	    return value
//	}
//	tCtx.Eventually(cb).Should(gomega.Equal(42), "should be the answer to everything")
//
// If there is no value, then an error can be returned:
//
//	cb := func(func(tCtx ktesting.TContext) error {
//	    err := doSomething(...)
//	    return err
//	}
//	tCtx.Eventually(cb).Should(gomega.Succeed(), "foobar should succeed")
//
// The default Gomega poll interval and timeout are used. Setting a specific
// timeout may be useful:
//
//	tCtx.Eventually(cb).Timeout(5 * time.Second).Should(gomega.Succeed(), "foobar should succeed")
//
// Canceling the context in the callback only affects code in the callback. The
// context passed to Eventually is not getting canceled. To abort polling
// immediately because the expected condition is known to not be reached
// anymore, use [gomega.StopTrying]:
//
//	cb := func(func(tCtx ktesting.TContext) int {
//	    value, err := doSomething(...)
//	    if errors.Is(err, SomeFinalErr) {
//	        // This message completely replaces the normal
//	        // failure message and thus should include all
//	        // relevant information.
//	        //
//	        // github.com/onsi/gomega/format is a good way
//	        // to format arbitrary data. It uses indention
//	        // and falls back to YAML for Kubernetes API
//	        // structs for readability.
//	        gomega.StopTrying("permanent failure, last value:\n%s", format.Object(value, 1 /* indent one level */)).
//	            Wrap(err).Now()
//	    }
//	    ktesting.ExpectNoError(tCtx, err, "something failed")
//	    return value
//	}
//	tCtx.Eventually(cb).Should(gomega.Equal(42), "should be the answer to everything")
//
// To poll again after some specific timeout, use [gomega.TryAgainAfter]. This is
// particularly useful in [Consistently] to ignore some intermittent error.
//
//	cb := func(func(tCtx ktesting.TContext) int {
//	    value, err := doSomething(...)
//	    var intermittentErr SomeIntermittentError
//	    if errors.As(err, &intermittentErr) {
//	        gomega.TryAgainAfter(intermittentErr.RetryPeriod).Wrap(err).Now()
//	    }
//	    ktesting.ExpectNoError(tCtx, err, "something failed")
//	    return value
//	 }
//	 tCtx.Eventually(cb).Should(gomega.Equal(42), "should be the answer to everything")
func Eventually[T any](tCtx TContext, cb func(TContext) T) gomega.AsyncAssertion {
	tCtx.Helper()
	return gomega.NewWithT(tCtx).Eventually(tCtx, func(ctx context.Context) (val T, err error) {
		tCtx := WithContext(tCtx, ctx)
		tCtx, finalize := WithError(tCtx, &err)
		defer finalize()
		tCtx = WithCancel(tCtx)
		return cb(tCtx), nil
	})
}

// Consistently wraps [gomega.Consistently] the same way as [Eventually] wraps
// [gomega.Eventually].
func Consistently[T any](tCtx TContext, cb func(TContext) T) gomega.AsyncAssertion {
	tCtx.Helper()
	return gomega.NewWithT(tCtx).Consistently(tCtx, func(ctx context.Context) (val T, err error) {
		tCtx := WithContext(tCtx, ctx)
		tCtx, finalize := WithError(tCtx, &err)
		defer finalize()
		return cb(tCtx), nil
	})
}
