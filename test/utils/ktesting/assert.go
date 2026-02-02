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
	"reflect"

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

// Eventually wraps [gomega.Eventually]. Supported argument types are:
//   - A function with a `tCtx ktesting.TContext` or `ctx context.Context`
//     parameter plus additional parameters and arbitrary return values.
//   - A value which changes over time, usually a channel.
//
// For functions, the context is provided by Eventually. Additional parameters
// must be passed via WithParameters. The first return value is passed to the Gomega
// matcher, all others (in particular an additional error) must be null.
// As a special case, a function with `tCtx ktesting.TContext` and no return
// values can be combined with gomega.Succeed as matcher.
//
// In contrast to direct usage of [gomega.Eventually], making additional
// assertions inside the callback function is okay in all cases as long as they
// use the TContext that is passed in. An assertion failure is considered
// temporary, so Eventually will continue to poll. This can be used to check a
// value with multiple assertions instead of writing a custom matcher:
//
//	cb := func(tCtx ktesting.TContext) {
//	    value, err := doSomething(...)
//	    tCtx.ExpectNoError(err, "something failed")
//	    tCtx.Assert(value.a).To(gomega.Equal(42), "the answer")
//	    tCtx.Assert(value.b).To(gomega.Equal("the fish"), "thanks")
//	}
//	tCtx.Eventually(cb).Should(gomega.Succeed())
//
// The test stops in case of a failure. To continue, use AssertEventually.
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
//	    tCtx.ExpectNoError(err, "something failed")
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
//	    tCtx.ExpectNoError(err, "something failed")
//	    return value
//	 }
//	 tCtx.Eventually(cb).Should(gomega.Equal(42), "should be the answer to everything")
func (tc *TC) Eventually(arg any) gomega.AsyncAssertion {
	tc.Helper()
	return tc.newAsyncAssertion(gomega.NewWithT(tc).Eventually, arg)
}

// AssertEventually is a variant of Eventually which merely records a failure
// without stopping the test.
func (tc *TC) AssertEventually(arg any) gomega.AsyncAssertion {
	tc.Helper()
	return tc.newAsyncAssertion(gomega.NewWithT(assertTestingT{tc}).Eventually, arg)
}

// Consistently wraps [gomega.Consistently] the same way as [Eventually] wraps
// [gomega.Eventually].
func (tc *TC) Consistently(arg any) gomega.AsyncAssertion {
	tc.Helper()
	return tc.newAsyncAssertion(gomega.NewWithT(tc).Consistently, arg)
}

// AssertConsistently is a variant of Consistently which merely records a failure
// without stopping the test.
func (tc *TC) AssertConsistently(arg any) gomega.AsyncAssertion {
	tc.Helper()
	return tc.newAsyncAssertion(gomega.NewWithT(assertTestingT{tc}).Consistently, arg)
}

func (tc *TC) newAsyncAssertion(eventuallyOrConsistently func(actualOrCtx any, args ...any) gomega.AsyncAssertion, arg any) gomega.AsyncAssertion {
	tc.Helper()
	// switch arg := arg.(type) {
	// case func(tCtx TContext):
	// 	// Tricky to handle via reflect, so let's cover this directly...
	// 	return eventuallyOrConsistently(tc, func(g gomega.Gomega, ctx context.Context) (err error) {
	// 		tCtx := WithContext(tc, ctx)
	// 		tCtx, finalize := WithError(tCtx, &err)
	// 		defer finalize()
	// 		arg(tCtx)
	// 	})
	// default:
	v := reflect.ValueOf(arg)
	if v.Kind() != reflect.Func {
		// Gomega must deal with it.
		return eventuallyOrConsistently(tc, arg)
	}
	t := v.Type()
	if t.NumIn() == 0 || t.In(0) != tContextType {
		// Not a function we can wrap.
		return eventuallyOrConsistently(tc, arg)
	}
	// Build a wrapper function with context instead of TContext as first parameter.
	// The wrapper then builds that TContext when called and invokes the actual function.
	in := make([]reflect.Type, t.NumIn())
	in[0] = contextType
	for i := 1; i < t.NumIn(); i++ {
		in[i] = t.In(i)
	}
	out := make([]reflect.Type, t.NumOut())
	for i := range t.NumOut() {
		out[i] = t.Out(i)
	}
	// The last result must always be an error because we need the ability to return assertion
	// failures, so we may have to add an error result value if the function doesn't
	// already have it.
	addErrResult := t.NumOut() == 0 || t.Out(t.NumOut()-1) != errorType
	if addErrResult {
		out = append(out, errorType)
	}
	wrapperType := reflect.FuncOf(in, out, t.IsVariadic())
	wrapper := reflect.MakeFunc(wrapperType, func(args []reflect.Value) (results []reflect.Value) {
		var err error
		tCtx, finalize := tc.WithContext(args[0].Interface().(context.Context)).
			WithCancel().
			WithError(&err)
		args[0] = reflect.ValueOf(tCtx)
		defer func() {
			// This runs *after* finalize.
			// If we are returning normally, then we must inject back the err
			// value that was set by finalize.
			if r := recover(); r != nil {
				// Nope, no results needed.
				panic(r)
			}
			errValue := reflect.ValueOf(err)
			if err == nil {
				// reflect doesn't like this ("returned zero Value").
				// We need a value of the right type.
				errValue = reflect.New(errorType).Elem()
			}
			// If the call panicked and the panic was recoved
			// by finalize(), then results is still nil.
			// We need to fill in null values.
			if len(results) == 0 && t.NumOut() > 0 {
				for i := range t.NumOut() {
					results = append(results, reflect.New(t.Out(i)).Elem())
				}
			}
			if addErrResult {
				results = append(results, errValue)
				return
			}
			if results[len(results)-1].IsNil() && err != nil {
				results[len(results)-1] = errValue
			}
		}()
		defer finalize() // Must be called directly, otherwise it cannot recover a panic.
		return v.Call(args)
	})
	return eventuallyOrConsistently(tc, wrapper.Interface())
}

var (
	contextType  = reflect.TypeFor[context.Context]()
	errorType    = reflect.TypeFor[error]()
	tContextType = reflect.TypeFor[TContext]()
)
