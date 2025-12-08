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

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/format"
	gtypes "github.com/onsi/gomega/types"
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

func gomegaAssertion(tc *TC, fatal bool, actual interface{}, extra ...interface{}) gomega.Assertion {
	testingT := gtypes.GomegaTestingT(tc)
	if !fatal {
		testingT = assertTestingT{tc}
	}
	return gomega.NewWithT(testingT).Expect(actual, extra...)
}

// assertTestingT implements Fatalf (the only function used by Gomega for
// reporting failures) using TContext.Errorf, i.e. testing continues after a
// failed assertion. The Helper method gets passed through.
type assertTestingT struct {
	*TC
}

var _ gtypes.GomegaTestingT = assertTestingT{}

func (a assertTestingT) Fatalf(format string, args ...any) {
	a.Helper()
	a.Errorf(format, args...)
}

// ExpectNoError asserts that no error has occurred and fails the test if it does.
//
// As in [gomega], the optional explanation can be:
//   - a [fmt.Sprintf] format string plus its arguments
//   - a function returning a string, which will be called
//     lazily to construct the explanation if needed
//
// If an explanation is provided, then it replaces the default "Unexpected
// error" in the failure message. It's combined with additional details by
// adding a colon at the end, as when wrapping an error. Therefore it should
// not end with a punctuation mark or line break.
//
// Using ExpectNoError instead of the corresponding Gomega or testify
// assertions has the advantage that the failure message is short (good for
// aggregation in https://go.k8s.io/triage) with more details captured in the
// test log output (good when investigating one particular failure).
//
// Helper packages should return errors that are derived from [FailureError].
// The test code then is forced to check for that error by the normal
// linter and should provide additional context for the failure, just
// as it would when printing or wrapping an error:
//
//	tCtx.ExpectNoError(somehelper.CreateSomething(tCtx, ...), "creating the first foobar")
//	tCtx.ExpectNoError(somehelper.CreateSomething(tCtx, ...), "creating the second foobar")
func (tc *TC) ExpectNoError(err error, explain ...interface{}) {
	tc.Helper()
	tc.noError(tc.Fatalf, err, explain...)
}

// AssertNoError is a variant of ExpectNoError which reports an unexpected
// error without aborting the test. It returns true if there was no error.
func (tc *TC) AssertNoError(err error, explain ...interface{}) bool {
	tc.Helper()
	return tc.noError(tc.Errorf, err, explain...)
}

func (tc *TC) noError(failf func(format string, args ...any), err error, explain ...interface{}) bool {
	if err == nil {
		return true
	}

	tc.Helper()
	description := buildDescription(explain...)

	if errors.Is(err, ErrFailure) {
		var failure FailureError
		if tc.capture == nil && errors.As(err, &failure) {
			if backtrace := failure.Backtrace(); backtrace != "" {
				if description != "" {
					tc.Log(description)
				}
				tc.Logf("Failed at:\n%s", backtrace)
			}
		}
		if description != "" {
			failf("%s: %s", description, err.Error())
			return false
		}
		failf("%s", err.Error())
		return false
	}

	if description == "" {
		description = "Unexpected error"
	}
	if tc.capture == nil {
		tc.Logf("%s:\n%s", description, format.Object(err, 0))
	}
	failf("%s: %v", description, err.Error())
	return false
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
