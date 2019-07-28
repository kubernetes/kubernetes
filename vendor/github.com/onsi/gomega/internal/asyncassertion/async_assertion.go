package asyncassertion

import (
	"errors"
	"fmt"
	"reflect"
	"time"

	"github.com/onsi/gomega/internal/oraclematcher"
	"github.com/onsi/gomega/types"
)

type AsyncAssertionType uint

const (
	AsyncAssertionTypeEventually AsyncAssertionType = iota
	AsyncAssertionTypeConsistently
)

type AsyncAssertion struct {
	asyncType       AsyncAssertionType
	actualInput     interface{}
	timeoutInterval time.Duration
	pollingInterval time.Duration
	failWrapper     *types.GomegaFailWrapper
	offset          int
}

func New(asyncType AsyncAssertionType, actualInput interface{}, failWrapper *types.GomegaFailWrapper, timeoutInterval time.Duration, pollingInterval time.Duration, offset int) *AsyncAssertion {
	actualType := reflect.TypeOf(actualInput)
	if actualType.Kind() == reflect.Func {
		if actualType.NumIn() != 0 || actualType.NumOut() == 0 {
			panic("Expected a function with no arguments and one or more return values.")
		}
	}

	return &AsyncAssertion{
		asyncType:       asyncType,
		actualInput:     actualInput,
		failWrapper:     failWrapper,
		timeoutInterval: timeoutInterval,
		pollingInterval: pollingInterval,
		offset:          offset,
	}
}

func (assertion *AsyncAssertion) Should(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool {
	assertion.failWrapper.TWithHelper.Helper()
	return assertion.match(matcher, true, optionalDescription...)
}

func (assertion *AsyncAssertion) ShouldNot(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool {
	assertion.failWrapper.TWithHelper.Helper()
	return assertion.match(matcher, false, optionalDescription...)
}

func (assertion *AsyncAssertion) buildDescription(optionalDescription ...interface{}) string {
	switch len(optionalDescription) {
	case 0:
		return ""
	default:
		return fmt.Sprintf(optionalDescription[0].(string), optionalDescription[1:]...) + "\n"
	}
}

func (assertion *AsyncAssertion) actualInputIsAFunction() bool {
	actualType := reflect.TypeOf(assertion.actualInput)
	return actualType.Kind() == reflect.Func && actualType.NumIn() == 0 && actualType.NumOut() > 0
}

func (assertion *AsyncAssertion) pollActual() (interface{}, error) {
	if assertion.actualInputIsAFunction() {
		values := reflect.ValueOf(assertion.actualInput).Call([]reflect.Value{})

		extras := []interface{}{}
		for _, value := range values[1:] {
			extras = append(extras, value.Interface())
		}

		success, message := vetExtras(extras)

		if !success {
			return nil, errors.New(message)
		}

		return values[0].Interface(), nil
	}

	return assertion.actualInput, nil
}

func (assertion *AsyncAssertion) matcherMayChange(matcher types.GomegaMatcher, value interface{}) bool {
	if assertion.actualInputIsAFunction() {
		return true
	}

	return oraclematcher.MatchMayChangeInTheFuture(matcher, value)
}

func (assertion *AsyncAssertion) match(matcher types.GomegaMatcher, desiredMatch bool, optionalDescription ...interface{}) bool {
	timer := time.Now()
	timeout := time.After(assertion.timeoutInterval)

	description := assertion.buildDescription(optionalDescription...)

	var matches bool
	var err error
	mayChange := true
	value, err := assertion.pollActual()
	if err == nil {
		mayChange = assertion.matcherMayChange(matcher, value)
		matches, err = matcher.Match(value)
	}

	assertion.failWrapper.TWithHelper.Helper()

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
		assertion.failWrapper.TWithHelper.Helper()
		assertion.failWrapper.Fail(fmt.Sprintf("%s after %.3fs.\n%s%s%s", preamble, time.Since(timer).Seconds(), description, message, errMsg), 3+assertion.offset)
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

func vetExtras(extras []interface{}) (bool, string) {
	for i, extra := range extras {
		if extra != nil {
			zeroValue := reflect.Zero(reflect.TypeOf(extra)).Interface()
			if !reflect.DeepEqual(zeroValue, extra) {
				message := fmt.Sprintf("Unexpected non-nil/non-zero extra argument at index %d:\n\t<%T>: %#v", i+1, extra, extra)
				return false, message
			}
		}
	}
	return true, ""
}
