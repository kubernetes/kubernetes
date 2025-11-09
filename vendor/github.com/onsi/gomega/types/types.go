package types

import (
	"context"
	"time"
)

type GomegaFailHandler func(message string, callerSkip ...int)

// A simple *testing.T interface wrapper
type GomegaTestingT interface {
	Helper()
	Fatalf(format string, args ...any)
}

// Gomega represents an object that can perform synchronous and asynchronous assertions with Gomega matchers
type Gomega interface {
	Ω(actual any, extra ...any) Assertion
	Expect(actual any, extra ...any) Assertion
	ExpectWithOffset(offset int, actual any, extra ...any) Assertion

	Eventually(actualOrCtx any, args ...any) AsyncAssertion
	EventuallyWithOffset(offset int, actualOrCtx any, args ...any) AsyncAssertion

	Consistently(actualOrCtx any, args ...any) AsyncAssertion
	ConsistentlyWithOffset(offset int, actualOrCtx any, args ...any) AsyncAssertion

	SetDefaultEventuallyTimeout(time.Duration)
	SetDefaultEventuallyPollingInterval(time.Duration)
	SetDefaultConsistentlyDuration(time.Duration)
	SetDefaultConsistentlyPollingInterval(time.Duration)
	EnforceDefaultTimeoutsWhenUsingContexts()
	DisableDefaultTimeoutsWhenUsingContext()
}

// All Gomega matchers must implement the GomegaMatcher interface
//
// For details on writing custom matchers, check out: http://onsi.github.io/gomega/#adding-your-own-matchers
type GomegaMatcher interface {
	Match(actual any) (success bool, err error)
	FailureMessage(actual any) (message string)
	NegatedFailureMessage(actual any) (message string)
}

/*
GomegaMatchers that also match the OracleMatcher interface can convey information about
whether or not their result will change upon future attempts.

This allows `Eventually` and `Consistently` to short circuit if success becomes impossible.

For example, a process' exit code can never change.  So, gexec's Exit matcher returns `true`
for `MatchMayChangeInTheFuture` until the process exits, at which point it returns `false` forevermore.
*/
type OracleMatcher interface {
	MatchMayChangeInTheFuture(actual any) bool
}

func MatchMayChangeInTheFuture(matcher GomegaMatcher, value any) bool {
	oracleMatcher, ok := matcher.(OracleMatcher)
	if !ok {
		return true
	}

	return oracleMatcher.MatchMayChangeInTheFuture(value)
}

// AsyncAssertions are returned by Eventually and Consistently and enable matchers to be polled repeatedly to ensure
// they are eventually satisfied
type AsyncAssertion interface {
	Should(matcher GomegaMatcher, optionalDescription ...any) bool
	ShouldNot(matcher GomegaMatcher, optionalDescription ...any) bool

	// equivalent to above
	To(matcher GomegaMatcher, optionalDescription ...any) bool
	ToNot(matcher GomegaMatcher, optionalDescription ...any) bool
	NotTo(matcher GomegaMatcher, optionalDescription ...any) bool

	WithOffset(offset int) AsyncAssertion
	WithTimeout(interval time.Duration) AsyncAssertion
	WithPolling(interval time.Duration) AsyncAssertion
	Within(timeout time.Duration) AsyncAssertion
	ProbeEvery(interval time.Duration) AsyncAssertion
	WithContext(ctx context.Context) AsyncAssertion
	WithArguments(argsToForward ...any) AsyncAssertion
	MustPassRepeatedly(count int) AsyncAssertion
}

// Assertions are returned by Ω and Expect and enable assertions against Gomega matchers
type Assertion interface {
	Should(matcher GomegaMatcher, optionalDescription ...any) bool
	ShouldNot(matcher GomegaMatcher, optionalDescription ...any) bool

	To(matcher GomegaMatcher, optionalDescription ...any) bool
	ToNot(matcher GomegaMatcher, optionalDescription ...any) bool
	NotTo(matcher GomegaMatcher, optionalDescription ...any) bool

	WithOffset(offset int) Assertion

	Error() Assertion
}
