package types

import (
	"context"
	"time"
)

type GomegaFailHandler func(message string, callerSkip ...int)

// A simple *testing.T interface wrapper
type GomegaTestingT interface {
	Helper()
	Fatalf(format string, args ...interface{})
}

// Gomega represents an object that can perform synchronous and assynchronous assertions with Gomega matchers
type Gomega interface {
	Ω(actual interface{}, extra ...interface{}) Assertion
	Expect(actual interface{}, extra ...interface{}) Assertion
	ExpectWithOffset(offset int, actual interface{}, extra ...interface{}) Assertion

	Eventually(actual interface{}, intervals ...interface{}) AsyncAssertion
	EventuallyWithOffset(offset int, actual interface{}, intervals ...interface{}) AsyncAssertion

	Consistently(actual interface{}, intervals ...interface{}) AsyncAssertion
	ConsistentlyWithOffset(offset int, actual interface{}, intervals ...interface{}) AsyncAssertion

	SetDefaultEventuallyTimeout(time.Duration)
	SetDefaultEventuallyPollingInterval(time.Duration)
	SetDefaultConsistentlyDuration(time.Duration)
	SetDefaultConsistentlyPollingInterval(time.Duration)
}

// All Gomega matchers must implement the GomegaMatcher interface
//
// For details on writing custom matchers, check out: http://onsi.github.io/gomega/#adding-your-own-matchers
type GomegaMatcher interface {
	Match(actual interface{}) (success bool, err error)
	FailureMessage(actual interface{}) (message string)
	NegatedFailureMessage(actual interface{}) (message string)
}

/*
GomegaMatchers that also match the OracleMatcher interface can convey information about
whether or not their result will change upon future attempts.

This allows `Eventually` and `Consistently` to short circuit if success becomes impossible.

For example, a process' exit code can never change.  So, gexec's Exit matcher returns `true`
for `MatchMayChangeInTheFuture` until the process exits, at which point it returns `false` forevermore.
*/
type OracleMatcher interface {
	MatchMayChangeInTheFuture(actual interface{}) bool
}

func MatchMayChangeInTheFuture(matcher GomegaMatcher, value interface{}) bool {
	oracleMatcher, ok := matcher.(OracleMatcher)
	if !ok {
		return true
	}

	return oracleMatcher.MatchMayChangeInTheFuture(value)
}

// AsyncAssertions are returned by Eventually and Consistently and enable matchers to be polled repeatedly to ensure
// they are eventually satisfied
type AsyncAssertion interface {
	Should(matcher GomegaMatcher, optionalDescription ...interface{}) bool
	ShouldNot(matcher GomegaMatcher, optionalDescription ...interface{}) bool

	WithOffset(offset int) AsyncAssertion
	WithTimeout(interval time.Duration) AsyncAssertion
	WithPolling(interval time.Duration) AsyncAssertion
	Within(timeout time.Duration) AsyncAssertion
	ProbeEvery(interval time.Duration) AsyncAssertion
	WithContext(ctx context.Context) AsyncAssertion
	WithArguments(argsToForward ...interface{}) AsyncAssertion
}

// Assertions are returned by Ω and Expect and enable assertions against Gomega matchers
type Assertion interface {
	Should(matcher GomegaMatcher, optionalDescription ...interface{}) bool
	ShouldNot(matcher GomegaMatcher, optionalDescription ...interface{}) bool

	To(matcher GomegaMatcher, optionalDescription ...interface{}) bool
	ToNot(matcher GomegaMatcher, optionalDescription ...interface{}) bool
	NotTo(matcher GomegaMatcher, optionalDescription ...interface{}) bool

	WithOffset(offset int) Assertion

	Error() Assertion
}
