package internal

import (
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"time"

	"github.com/onsi/gomega/types"
)

type AsyncAssertionType uint

const (
	AsyncAssertionTypeEventually AsyncAssertionType = iota
	AsyncAssertionTypeConsistently
)

type AsyncAssertion struct {
	asyncType AsyncAssertionType

	actualIsFunc bool
	actualValue  interface{}
	actualFunc   func() ([]reflect.Value, error)

	timeoutInterval time.Duration
	pollingInterval time.Duration
	offset          int
	g               *Gomega
}

func NewAsyncAssertion(asyncType AsyncAssertionType, actualInput interface{}, g *Gomega, timeoutInterval time.Duration, pollingInterval time.Duration, offset int) *AsyncAssertion {
	out := &AsyncAssertion{
		asyncType:       asyncType,
		timeoutInterval: timeoutInterval,
		pollingInterval: pollingInterval,
		offset:          offset,
		g:               g,
	}

	switch actualType := reflect.TypeOf(actualInput); {
	case actualType.Kind() != reflect.Func:
		out.actualValue = actualInput
	case actualType.NumIn() == 0 && actualType.NumOut() > 0:
		out.actualIsFunc = true
		out.actualFunc = func() ([]reflect.Value, error) {
			return reflect.ValueOf(actualInput).Call([]reflect.Value{}), nil
		}
	case actualType.NumIn() == 1 && actualType.In(0).Implements(reflect.TypeOf((*types.Gomega)(nil)).Elem()):
		out.actualIsFunc = true
		out.actualFunc = func() (values []reflect.Value, err error) {
			var assertionFailure error
			assertionCapturingGomega := NewGomega(g.DurationBundle).ConfigureWithFailHandler(func(message string, callerSkip ...int) {
				skip := 0
				if len(callerSkip) > 0 {
					skip = callerSkip[0]
				}
				_, file, line, _ := runtime.Caller(skip + 1)
				assertionFailure = fmt.Errorf("Assertion in callback at %s:%d failed:\n%s", file, line, message)
				panic("stop execution")
			})

			defer func() {
				if actualType.NumOut() == 0 {
					if assertionFailure == nil {
						values = []reflect.Value{reflect.Zero(reflect.TypeOf((*error)(nil)).Elem())}
					} else {
						values = []reflect.Value{reflect.ValueOf(assertionFailure)}
					}
				} else {
					err = assertionFailure
				}
				if e := recover(); e != nil && assertionFailure == nil {
					panic(e)
				}
			}()

			values = reflect.ValueOf(actualInput).Call([]reflect.Value{reflect.ValueOf(assertionCapturingGomega)})
			return
		}
	default:
		msg := fmt.Sprintf("The function passed to Gomega's async assertions should either take no arguments and return values, or take a single Gomega interface that it can use to make assertions within the body of the function.  When taking a Gomega interface the function can optionally return values or return nothing.  The function you passed takes %d arguments and returns %d values.", actualType.NumIn(), actualType.NumOut())
		g.Fail(msg, offset+4)
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

func (assertion *AsyncAssertion) Should(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool {
	assertion.g.THelper()
	return assertion.match(matcher, true, optionalDescription...)
}

func (assertion *AsyncAssertion) ShouldNot(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool {
	assertion.g.THelper()
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

func (assertion *AsyncAssertion) pollActual() (interface{}, error) {
	if !assertion.actualIsFunc {
		return assertion.actualValue, nil
	}

	values, err := assertion.actualFunc()
	if err != nil {
		return nil, err
	}
	extras := []interface{}{nil}
	for _, value := range values[1:] {
		extras = append(extras, value.Interface())
	}
	success, message := vetActuals(extras, 0)
	if !success {
		return nil, errors.New(message)
	}

	return values[0].Interface(), nil
}

func (assertion *AsyncAssertion) matcherMayChange(matcher types.GomegaMatcher, value interface{}) bool {
	if assertion.actualIsFunc {
		return true
	}
	return types.MatchMayChangeInTheFuture(matcher, value)
}

func (assertion *AsyncAssertion) match(matcher types.GomegaMatcher, desiredMatch bool, optionalDescription ...interface{}) bool {
	timer := time.Now()
	timeout := time.After(assertion.timeoutInterval)

	var matches bool
	var err error
	mayChange := true
	value, err := assertion.pollActual()
	if err == nil {
		mayChange = assertion.matcherMayChange(matcher, value)
		matches, err = matcher.Match(value)
	}

	assertion.g.THelper()

	fail := func(preamble string) {
		errMsg := ""
		message := ""
		if err != nil {
			errMsg = "Error: " + err.Error()
		} else {
			if desiredMatch {
				message = matcher.FailureMessage(value)
			} else {
				message = matcher.NegatedFailureMessage(value)
			}
		}
		assertion.g.THelper()
		description := assertion.buildDescription(optionalDescription...)
		assertion.g.Fail(fmt.Sprintf("%s after %.3fs.\n%s%s%s", preamble, time.Since(timer).Seconds(), description, message, errMsg), 3+assertion.offset)
	}

	if assertion.asyncType == AsyncAssertionTypeEventually {
		for {
			if err == nil && matches == desiredMatch {
				return true
			}

			if !mayChange {
				fail("No future change is possible.  Bailing out early")
				return false
			}

			select {
			case <-time.After(assertion.pollingInterval):
				value, err = assertion.pollActual()
				if err == nil {
					mayChange = assertion.matcherMayChange(matcher, value)
					matches, err = matcher.Match(value)
				}
			case <-timeout:
				fail("Timed out")
				return false
			}
		}
	} else if assertion.asyncType == AsyncAssertionTypeConsistently {
		for {
			if !(err == nil && matches == desiredMatch) {
				fail("Failed")
				return false
			}

			if !mayChange {
				return true
			}

			select {
			case <-time.After(assertion.pollingInterval):
				value, err = assertion.pollActual()
				if err == nil {
					mayChange = assertion.matcherMayChange(matcher, value)
					matches, err = matcher.Match(value)
				}
			case <-timeout:
				return true
			}
		}
	}

	return false
}
