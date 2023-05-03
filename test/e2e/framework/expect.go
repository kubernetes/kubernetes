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

package framework

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	ginkgotypes "github.com/onsi/ginkgo/v2/types"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/types"
)

// MakeMatcher builds a gomega.Matcher based on a single callback function.
// That function is passed the actual value that is to be checked.
// There are three possible outcomes of the check:
//   - An error is returned, which then is converted into a failure
//     by Gomega.
//   - A non-nil failure function is returned, which then is called
//     by Gomega once a failure string is needed. This is useful
//     to avoid unnecessarily preparing a failure string for intermediate
//     failures in Eventually or Consistently.
//   - Both function and error are nil, which means that the check
//     succeeded.
func MakeMatcher[T interface{}](match func(actual T) (failure func() string, err error)) types.GomegaMatcher {
	return &matcher[T]{
		match: match,
	}
}

type matcher[T interface{}] struct {
	match   func(actual T) (func() string, error)
	failure func() string
}

func (m *matcher[T]) Match(actual interface{}) (success bool, err error) {
	if actual, ok := actual.(T); ok {
		failure, err := m.match(actual)
		if err != nil {
			return false, err
		}
		m.failure = failure
		if failure != nil {
			return false, nil
		}
		return true, nil
	}
	var empty T
	return false, gomega.StopTrying(fmt.Sprintf("internal error: expected %T, got:\n%s", empty, format.Object(actual, 1)))
}

func (m *matcher[T]) FailureMessage(actual interface{}) string {
	return m.failure()
}

func (m matcher[T]) NegatedFailureMessage(actual interface{}) string {
	return m.failure()
}

var _ types.GomegaMatcher = &matcher[string]{}

// Gomega returns an interface that can be used like gomega to express
// assertions. The difference is that failed assertions are returned as an
// error:
//
//	if err := Gomega().Expect(pod.Status.Phase).To(gomega.BeEqual(v1.Running)); err != nil {
//	    return fmt.Errorf("test pod not running: %w", err)
//	}
//
// This error can get wrapped to provide additional context for the
// failure. The test then should use ExpectNoError to turn a non-nil error into
// a failure.
//
// When using this approach, there is no need for call offsets and extra
// descriptions for the Expect call because the call stack will be dumped when
// ExpectNoError is called and the additional description(s) can be added by
// wrapping the error.
//
// Asynchronous assertions use the framework's Poll interval and PodStart timeout
// by default.
func Gomega() GomegaInstance {
	return gomegaInstance{}
}

type GomegaInstance interface {
	Expect(actual interface{}) Assertion
	Eventually(ctx context.Context, args ...interface{}) AsyncAssertion
	Consistently(ctx context.Context, args ...interface{}) AsyncAssertion
}

type Assertion interface {
	Should(matcher types.GomegaMatcher) error
	ShouldNot(matcher types.GomegaMatcher) error
	To(matcher types.GomegaMatcher) error
	ToNot(matcher types.GomegaMatcher) error
	NotTo(matcher types.GomegaMatcher) error
}

type AsyncAssertion interface {
	Should(matcher types.GomegaMatcher) error
	ShouldNot(matcher types.GomegaMatcher) error

	WithTimeout(interval time.Duration) AsyncAssertion
	WithPolling(interval time.Duration) AsyncAssertion
}

type gomegaInstance struct{}

var _ GomegaInstance = gomegaInstance{}

func (g gomegaInstance) Expect(actual interface{}) Assertion {
	return assertion{actual: actual}
}

func (g gomegaInstance) Eventually(ctx context.Context, args ...interface{}) AsyncAssertion {
	return newAsyncAssertion(ctx, args, false)
}

func (g gomegaInstance) Consistently(ctx context.Context, args ...interface{}) AsyncAssertion {
	return newAsyncAssertion(ctx, args, true)
}

func newG() (*FailureError, gomega.Gomega) {
	var failure FailureError
	g := gomega.NewGomega(func(msg string, callerSkip ...int) {
		failure = FailureError{
			msg: msg,
		}
	})

	return &failure, g
}

type assertion struct {
	actual interface{}
}

func (a assertion) Should(matcher types.GomegaMatcher) error {
	err, g := newG()
	if !g.Expect(a.actual).Should(matcher) {
		err.backtrace()
		return *err
	}
	return nil
}

func (a assertion) ShouldNot(matcher types.GomegaMatcher) error {
	err, g := newG()
	if !g.Expect(a.actual).ShouldNot(matcher) {
		err.backtrace()
		return *err
	}
	return nil
}

func (a assertion) To(matcher types.GomegaMatcher) error {
	err, g := newG()
	if !g.Expect(a.actual).To(matcher) {
		err.backtrace()
		return *err
	}
	return nil
}

func (a assertion) ToNot(matcher types.GomegaMatcher) error {
	err, g := newG()
	if !g.Expect(a.actual).ToNot(matcher) {
		err.backtrace()
		return *err
	}
	return nil
}

func (a assertion) NotTo(matcher types.GomegaMatcher) error {
	err, g := newG()
	if !g.Expect(a.actual).NotTo(matcher) {
		err.backtrace()
		return *err
	}
	return nil
}

type asyncAssertion struct {
	ctx          context.Context
	args         []interface{}
	timeout      time.Duration
	interval     time.Duration
	consistently bool
}

func newAsyncAssertion(ctx context.Context, args []interface{}, consistently bool) asyncAssertion {
	return asyncAssertion{
		ctx:  ctx,
		args: args,
		// PodStart is used as default because waiting for a pod is the
		// most common operation.
		timeout:  TestContext.timeouts.PodStart,
		interval: TestContext.timeouts.Poll,
	}
}

func (a asyncAssertion) newAsync() (*FailureError, gomega.AsyncAssertion) {
	err, g := newG()
	var assertion gomega.AsyncAssertion
	if a.consistently {
		assertion = g.Consistently(a.ctx, a.args...)
	} else {
		assertion = g.Eventually(a.ctx, a.args...)
	}
	assertion = assertion.WithTimeout(a.timeout).WithPolling(a.interval)
	return err, assertion
}

func (a asyncAssertion) Should(matcher types.GomegaMatcher) error {
	err, assertion := a.newAsync()
	if !assertion.Should(matcher) {
		err.backtrace()
		return *err
	}
	return nil
}

func (a asyncAssertion) ShouldNot(matcher types.GomegaMatcher) error {
	err, assertion := a.newAsync()
	if !assertion.ShouldNot(matcher) {
		err.backtrace()
		return *err
	}
	return nil
}

func (a asyncAssertion) WithTimeout(timeout time.Duration) AsyncAssertion {
	a.timeout = timeout
	return a
}

func (a asyncAssertion) WithPolling(interval time.Duration) AsyncAssertion {
	a.interval = interval
	return a
}

// FailureError is an error where the error string is meant to be passed to
// ginkgo.Fail directly, i.e. adding some prefix like "unexpected error" is not
// necessary. It is also not necessary to dump the error struct.
type FailureError struct {
	msg            string
	fullStackTrace string
}

func (f FailureError) Error() string {
	return f.msg
}

func (f FailureError) Backtrace() string {
	return f.fullStackTrace
}

func (f FailureError) Is(target error) bool {
	return target == ErrFailure
}

func (f *FailureError) backtrace() {
	f.fullStackTrace = ginkgotypes.NewCodeLocationWithStackTrace(2).FullStackTrace
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

// ExpectEqual expects the specified two are the same, otherwise an exception raises
//
// Deprecated: use gomega.Expect().To(gomega.BeEqual())
func ExpectEqual(actual interface{}, extra interface{}, explain ...interface{}) {
	gomega.ExpectWithOffset(1, actual).To(gomega.Equal(extra), explain...)
}

// ExpectNotEqual expects the specified two are not the same, otherwise an exception raises
//
// Deprecated: use gomega.Expect().ToNot(gomega.BeEqual())
func ExpectNotEqual(actual interface{}, extra interface{}, explain ...interface{}) {
	gomega.ExpectWithOffset(1, actual).NotTo(gomega.Equal(extra), explain...)
}

// ExpectError expects an error happens, otherwise an exception raises
//
// Deprecated: use gomega.Expect().To(gomega.HaveOccurred()) or (better!) check
// specifically for the error that is expected with
// gomega.Expect().To(gomega.MatchError(gomega.ContainSubstring()))
func ExpectError(err error, explain ...interface{}) {
	gomega.ExpectWithOffset(1, err).To(gomega.HaveOccurred(), explain...)
}

// ExpectNoError checks if "err" is set, and if so, fails assertion while logging the error.
func ExpectNoError(err error, explain ...interface{}) {
	ExpectNoErrorWithOffset(1, err, explain...)
}

// ExpectNoErrorWithOffset checks if "err" is set, and if so, fails assertion while logging the error at "offset" levels above its caller
// (for example, for call chain f -> g -> ExpectNoErrorWithOffset(1, ...) error would be logged for "f").
func ExpectNoErrorWithOffset(offset int, err error, explain ...interface{}) {
	if err == nil {
		return
	}

	// Errors usually contain unexported fields. We have to use
	// a formatter here which can print those.
	prefix := ""
	if len(explain) > 0 {
		if str, ok := explain[0].(string); ok {
			prefix = fmt.Sprintf(str, explain[1:]...) + ": "
		} else {
			prefix = fmt.Sprintf("unexpected explain arguments, need format string: %v", explain)
		}
	}

	// This intentionally doesn't use gomega.Expect. Instead we take
	// full control over what information is presented where:
	// - The complete error object is logged because it may contain
	//   additional information that isn't included in its error
	//   string.
	// - It is not included in the failure message because
	//   it might make the failure message very large and/or
	//   cause error aggregation to work less well: two
	//   failures at the same code line might not be matched in
	//   https://go.k8s.io/triage because the error details are too
	//   different.
	//
	// Some errors include all relevant information in the Error
	// string. For those we can skip the redundant log message.
	// For our own failures we only log the additional stack backtrace
	// because it is not included in the failure message.
	var failure FailureError
	if errors.As(err, &failure) && failure.Backtrace() != "" {
		Logf("Failed inside E2E framework:\n    %s", strings.ReplaceAll(failure.Backtrace(), "\n", "\n    "))
	} else if !errors.Is(err, ErrFailure) {
		Logf("Unexpected error: %s\n%s", prefix, format.Object(err, 1))
	}
	Fail(prefix+err.Error(), 1+offset)
}

// ExpectConsistOf expects actual contains precisely the extra elements.  The ordering of the elements does not matter.
//
// Deprecated: use gomega.Expect().To(gomega.ConsistOf()) instead
func ExpectConsistOf(actual interface{}, extra interface{}, explain ...interface{}) {
	gomega.ExpectWithOffset(1, actual).To(gomega.ConsistOf(extra), explain...)
}

// ExpectHaveKey expects the actual map has the key in the keyset
//
// Deprecated: use gomega.Expect().To(gomega.HaveKey()) instead
func ExpectHaveKey(actual interface{}, key interface{}, explain ...interface{}) {
	gomega.ExpectWithOffset(1, actual).To(gomega.HaveKey(key), explain...)
}

// ExpectEmpty expects actual is empty
//
// Deprecated: use gomega.Expect().To(gomega.BeEmpty()) instead
func ExpectEmpty(actual interface{}, explain ...interface{}) {
	gomega.ExpectWithOffset(1, actual).To(gomega.BeEmpty(), explain...)
}
