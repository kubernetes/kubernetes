package internal

import (
	"context"
	"fmt"
	"reflect"
	"runtime"
	"sync"
	"time"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/types"
)

var errInterface = reflect.TypeOf((*error)(nil)).Elem()
var gomegaType = reflect.TypeOf((*types.Gomega)(nil)).Elem()
var contextType = reflect.TypeOf(new(context.Context)).Elem()

type contextWithAttachProgressReporter interface {
	AttachProgressReporter(func() string) func()
}

type asyncGomegaHaltExecutionError struct{}

func (a asyncGomegaHaltExecutionError) GinkgoRecoverShouldIgnoreThisPanic() {}
func (a asyncGomegaHaltExecutionError) Error() string {
	return `An assertion has failed in a goroutine.  You should call 

    defer GinkgoRecover()

at the top of the goroutine that caused this panic.  This will allow Ginkgo and Gomega to correctly capture and manage this panic.`
}

type AsyncAssertionType uint

const (
	AsyncAssertionTypeEventually AsyncAssertionType = iota
	AsyncAssertionTypeConsistently
)

func (at AsyncAssertionType) String() string {
	switch at {
	case AsyncAssertionTypeEventually:
		return "Eventually"
	case AsyncAssertionTypeConsistently:
		return "Consistently"
	}
	return "INVALID ASYNC ASSERTION TYPE"
}

type AsyncAssertion struct {
	asyncType AsyncAssertionType

	actualIsFunc  bool
	actual        interface{}
	argsToForward []interface{}

	timeoutInterval time.Duration
	pollingInterval time.Duration
	ctx             context.Context
	offset          int
	g               *Gomega
}

func NewAsyncAssertion(asyncType AsyncAssertionType, actualInput interface{}, g *Gomega, timeoutInterval time.Duration, pollingInterval time.Duration, ctx context.Context, offset int) *AsyncAssertion {
	out := &AsyncAssertion{
		asyncType:       asyncType,
		timeoutInterval: timeoutInterval,
		pollingInterval: pollingInterval,
		offset:          offset,
		ctx:             ctx,
		g:               g,
	}

	out.actual = actualInput
	if actualInput != nil && reflect.TypeOf(actualInput).Kind() == reflect.Func {
		out.actualIsFunc = true
	}

	return out
}

func (assertion *AsyncAssertion) WithOffset(offset int) types.AsyncAssertion {
	assertion.offset = offset
	return assertion
}

func (assertion *AsyncAssertion) WithTimeout(interval time.Duration) types.AsyncAssertion {
	assertion.timeoutInterval = interval
	return assertion
}

func (assertion *AsyncAssertion) WithPolling(interval time.Duration) types.AsyncAssertion {
	assertion.pollingInterval = interval
	return assertion
}

func (assertion *AsyncAssertion) Within(timeout time.Duration) types.AsyncAssertion {
	assertion.timeoutInterval = timeout
	return assertion
}

func (assertion *AsyncAssertion) ProbeEvery(interval time.Duration) types.AsyncAssertion {
	assertion.pollingInterval = interval
	return assertion
}

func (assertion *AsyncAssertion) WithContext(ctx context.Context) types.AsyncAssertion {
	assertion.ctx = ctx
	return assertion
}

func (assertion *AsyncAssertion) WithArguments(argsToForward ...interface{}) types.AsyncAssertion {
	assertion.argsToForward = argsToForward
	return assertion
}

func (assertion *AsyncAssertion) Should(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool {
	assertion.g.THelper()
	vetOptionalDescription("Asynchronous assertion", optionalDescription...)
	return assertion.match(matcher, true, optionalDescription...)
}

func (assertion *AsyncAssertion) ShouldNot(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool {
	assertion.g.THelper()
	vetOptionalDescription("Asynchronous assertion", optionalDescription...)
	return assertion.match(matcher, false, optionalDescription...)
}

func (assertion *AsyncAssertion) buildDescription(optionalDescription ...interface{}) string {
	switch len(optionalDescription) {
	case 0:
		return ""
	case 1:
		if describe, ok := optionalDescription[0].(func() string); ok {
			return describe() + "\n"
		}
	}
	return fmt.Sprintf(optionalDescription[0].(string), optionalDescription[1:]...) + "\n"
}

func (assertion *AsyncAssertion) processReturnValues(values []reflect.Value) (interface{}, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("No values were returned by the function passed to Gomega")
	}

	actual := values[0].Interface()
	if _, ok := AsPollingSignalError(actual); ok {
		return actual, actual.(error)
	}

	var err error
	for i, extraValue := range values[1:] {
		extra := extraValue.Interface()
		if extra == nil {
			continue
		}
		if _, ok := AsPollingSignalError(extra); ok {
			return actual, extra.(error)
		}
		extraType := reflect.TypeOf(extra)
		zero := reflect.Zero(extraType).Interface()
		if reflect.DeepEqual(extra, zero) {
			continue
		}
		if i == len(values)-2 && extraType.Implements(errInterface) {
			err = fmt.Errorf("function returned error: %w", extra.(error))
		}
		if err == nil {
			err = fmt.Errorf("Unexpected non-nil/non-zero return value at index %d:\n\t<%T>: %#v", i+1, extra, extra)
		}
	}

	return actual, err
}

func (assertion *AsyncAssertion) invalidFunctionError(t reflect.Type) error {
	return fmt.Errorf(`The function passed to %s had an invalid signature of %s.  Functions passed to %s must either:

	(a) have return values or
	(b) take a Gomega interface as their first argument and use that Gomega instance to make assertions.

You can learn more at https://onsi.github.io/gomega/#eventually
`, assertion.asyncType, t, assertion.asyncType)
}

func (assertion *AsyncAssertion) noConfiguredContextForFunctionError() error {
	return fmt.Errorf(`The function passed to %s requested a context.Context, but no context has been provided.  Please pass one in using %s().WithContext().

You can learn more at https://onsi.github.io/gomega/#eventually
`, assertion.asyncType, assertion.asyncType)
}

func (assertion *AsyncAssertion) argumentMismatchError(t reflect.Type, numProvided int) error {
	have := "have"
	if numProvided == 1 {
		have = "has"
	}
	return fmt.Errorf(`The function passed to %s has signature %s takes %d arguments but %d %s been provided.  Please use %s().WithArguments() to pass the corect set of arguments.

You can learn more at https://onsi.github.io/gomega/#eventually
`, assertion.asyncType, t, t.NumIn(), numProvided, have, assertion.asyncType)
}

func (assertion *AsyncAssertion) buildActualPoller() (func() (interface{}, error), error) {
	if !assertion.actualIsFunc {
		return func() (interface{}, error) { return assertion.actual, nil }, nil
	}
	actualValue := reflect.ValueOf(assertion.actual)
	actualType := reflect.TypeOf(assertion.actual)
	numIn, numOut, isVariadic := actualType.NumIn(), actualType.NumOut(), actualType.IsVariadic()

	if numIn == 0 && numOut == 0 {
		return nil, assertion.invalidFunctionError(actualType)
	}
	takesGomega, takesContext := false, false
	if numIn > 0 {
		takesGomega, takesContext = actualType.In(0).Implements(gomegaType), actualType.In(0).Implements(contextType)
	}
	if takesGomega && numIn > 1 && actualType.In(1).Implements(contextType) {
		takesContext = true
	}
	if takesContext && len(assertion.argsToForward) > 0 && reflect.TypeOf(assertion.argsToForward[0]).Implements(contextType) {
		takesContext = false
	}
	if !takesGomega && numOut == 0 {
		return nil, assertion.invalidFunctionError(actualType)
	}
	if takesContext && assertion.ctx == nil {
		return nil, assertion.noConfiguredContextForFunctionError()
	}

	var assertionFailure error
	inValues := []reflect.Value{}
	if takesGomega {
		inValues = append(inValues, reflect.ValueOf(NewGomega(assertion.g.DurationBundle).ConfigureWithFailHandler(func(message string, callerSkip ...int) {
			skip := 0
			if len(callerSkip) > 0 {
				skip = callerSkip[0]
			}
			_, file, line, _ := runtime.Caller(skip + 1)
			assertionFailure = fmt.Errorf("Assertion in callback at %s:%d failed:\n%s", file, line, message)
			// we throw an asyncGomegaHaltExecutionError so that defer GinkgoRecover() can catch this error if the user makes an assertion in a goroutine
			panic(asyncGomegaHaltExecutionError{})
		})))
	}
	if takesContext {
		inValues = append(inValues, reflect.ValueOf(assertion.ctx))
	}
	for _, arg := range assertion.argsToForward {
		inValues = append(inValues, reflect.ValueOf(arg))
	}

	if !isVariadic && numIn != len(inValues) {
		return nil, assertion.argumentMismatchError(actualType, len(inValues))
	} else if isVariadic && len(inValues) < numIn-1 {
		return nil, assertion.argumentMismatchError(actualType, len(inValues))
	}

	return func() (actual interface{}, err error) {
		var values []reflect.Value
		assertionFailure = nil
		defer func() {
			if numOut == 0 && takesGomega {
				actual = assertionFailure
			} else {
				actual, err = assertion.processReturnValues(values)
				_, isAsyncError := AsPollingSignalError(err)
				if assertionFailure != nil && !isAsyncError {
					err = assertionFailure
				}
			}
			if e := recover(); e != nil {
				if _, isAsyncError := AsPollingSignalError(e); isAsyncError {
					err = e.(error)
				} else if assertionFailure == nil {
					panic(e)
				}
			}
		}()
		values = actualValue.Call(inValues)
		return
	}, nil
}

func (assertion *AsyncAssertion) afterTimeout() <-chan time.Time {
	if assertion.timeoutInterval >= 0 {
		return time.After(assertion.timeoutInterval)
	}

	if assertion.asyncType == AsyncAssertionTypeConsistently {
		return time.After(assertion.g.DurationBundle.ConsistentlyDuration)
	} else {
		if assertion.ctx == nil {
			return time.After(assertion.g.DurationBundle.EventuallyTimeout)
		} else {
			return nil
		}
	}
}

func (assertion *AsyncAssertion) afterPolling() <-chan time.Time {
	if assertion.pollingInterval >= 0 {
		return time.After(assertion.pollingInterval)
	}
	if assertion.asyncType == AsyncAssertionTypeConsistently {
		return time.After(assertion.g.DurationBundle.ConsistentlyPollingInterval)
	} else {
		return time.After(assertion.g.DurationBundle.EventuallyPollingInterval)
	}
}

func (assertion *AsyncAssertion) matcherSaysStopTrying(matcher types.GomegaMatcher, value interface{}) bool {
	if assertion.actualIsFunc || types.MatchMayChangeInTheFuture(matcher, value) {
		return false
	}
	return true
}

func (assertion *AsyncAssertion) pollMatcher(matcher types.GomegaMatcher, value interface{}) (matches bool, err error) {
	defer func() {
		if e := recover(); e != nil {
			if _, isAsyncError := AsPollingSignalError(e); isAsyncError {
				err = e.(error)
			} else {
				panic(e)
			}
		}
	}()

	matches, err = matcher.Match(value)

	return
}

func (assertion *AsyncAssertion) match(matcher types.GomegaMatcher, desiredMatch bool, optionalDescription ...interface{}) bool {
	timer := time.Now()
	timeout := assertion.afterTimeout()
	lock := sync.Mutex{}

	var matches bool
	var err error
	var oracleMatcherSaysStop bool

	assertion.g.THelper()

	pollActual, err := assertion.buildActualPoller()
	if err != nil {
		assertion.g.Fail(err.Error(), 2+assertion.offset)
		return false
	}

	value, err := pollActual()
	if err == nil {
		oracleMatcherSaysStop = assertion.matcherSaysStopTrying(matcher, value)
		matches, err = assertion.pollMatcher(matcher, value)
	}

	messageGenerator := func() string {
		// can be called out of band by Ginkgo if the user requests a progress report
		lock.Lock()
		defer lock.Unlock()
		message := ""
		if err != nil {
			if pollingSignalErr, ok := AsPollingSignalError(err); ok && pollingSignalErr.IsStopTrying() {
				message = err.Error()
				for _, attachment := range pollingSignalErr.Attachments {
					message += fmt.Sprintf("\n%s:\n", attachment.Description)
					message += format.Object(attachment.Object, 1)
				}
			} else {
				message = "Error: " + err.Error() + "\n" + format.Object(err, 1)
			}
		} else {
			if desiredMatch {
				message = matcher.FailureMessage(value)
			} else {
				message = matcher.NegatedFailureMessage(value)
			}
		}
		description := assertion.buildDescription(optionalDescription...)
		return fmt.Sprintf("%s%s", description, message)
	}

	fail := func(preamble string) {
		assertion.g.THelper()
		assertion.g.Fail(fmt.Sprintf("%s after %.3fs.\n%s", preamble, time.Since(timer).Seconds(), messageGenerator()), 3+assertion.offset)
	}

	var contextDone <-chan struct{}
	if assertion.ctx != nil {
		contextDone = assertion.ctx.Done()
		if v, ok := assertion.ctx.Value("GINKGO_SPEC_CONTEXT").(contextWithAttachProgressReporter); ok {
			detach := v.AttachProgressReporter(messageGenerator)
			defer detach()
		}
	}

	for {
		var nextPoll <-chan time.Time = nil
		var isTryAgainAfterError = false

		if pollingSignalErr, ok := AsPollingSignalError(err); ok {
			if pollingSignalErr.IsStopTrying() {
				fail("Told to stop trying")
				return false
			}
			if pollingSignalErr.IsTryAgainAfter() {
				nextPoll = time.After(pollingSignalErr.TryAgainDuration())
				isTryAgainAfterError = true
			}
		}

		if err == nil && matches == desiredMatch {
			if assertion.asyncType == AsyncAssertionTypeEventually {
				return true
			}
		} else if !isTryAgainAfterError {
			if assertion.asyncType == AsyncAssertionTypeConsistently {
				fail("Failed")
				return false
			}
		}

		if oracleMatcherSaysStop {
			if assertion.asyncType == AsyncAssertionTypeEventually {
				fail("No future change is possible.  Bailing out early")
				return false
			} else {
				return true
			}
		}

		if nextPoll == nil {
			nextPoll = assertion.afterPolling()
		}

		select {
		case <-nextPoll:
			v, e := pollActual()
			lock.Lock()
			value, err = v, e
			lock.Unlock()
			if err == nil {
				oracleMatcherSaysStop = assertion.matcherSaysStopTrying(matcher, value)
				m, e := assertion.pollMatcher(matcher, value)
				lock.Lock()
				matches, err = m, e
				lock.Unlock()
			}
		case <-contextDone:
			fail("Context was cancelled")
			return false
		case <-timeout:
			if assertion.asyncType == AsyncAssertionTypeEventually {
				fail("Timed out")
				return false
			} else {
				if isTryAgainAfterError {
					fail("Timed out while waiting on TryAgainAfter")
					return false
				}
				return true
			}
		}
	}
}
