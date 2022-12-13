package internal

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/gomega/types"
)

type Gomega struct {
	Fail           types.GomegaFailHandler
	THelper        func()
	DurationBundle DurationBundle
}

func NewGomega(bundle DurationBundle) *Gomega {
	return &Gomega{
		Fail:           nil,
		THelper:        nil,
		DurationBundle: bundle,
	}
}

func (g *Gomega) IsConfigured() bool {
	return g.Fail != nil && g.THelper != nil
}

func (g *Gomega) ConfigureWithFailHandler(fail types.GomegaFailHandler) *Gomega {
	g.Fail = fail
	g.THelper = func() {}
	return g
}

func (g *Gomega) ConfigureWithT(t types.GomegaTestingT) *Gomega {
	g.Fail = func(message string, _ ...int) {
		t.Helper()
		t.Fatalf("\n%s", message)
	}
	g.THelper = t.Helper
	return g
}

func (g *Gomega) Ω(actual interface{}, extra ...interface{}) types.Assertion {
	return g.ExpectWithOffset(0, actual, extra...)
}

func (g *Gomega) Expect(actual interface{}, extra ...interface{}) types.Assertion {
	return g.ExpectWithOffset(0, actual, extra...)
}

func (g *Gomega) ExpectWithOffset(offset int, actual interface{}, extra ...interface{}) types.Assertion {
	return NewAssertion(actual, g, offset, extra...)
}

func (g *Gomega) Eventually(args ...interface{}) types.AsyncAssertion {
	return g.makeAsyncAssertion(AsyncAssertionTypeEventually, 0, args...)
}

func (g *Gomega) EventuallyWithOffset(offset int, args ...interface{}) types.AsyncAssertion {
	return g.makeAsyncAssertion(AsyncAssertionTypeEventually, offset, args...)
}

func (g *Gomega) Consistently(args ...interface{}) types.AsyncAssertion {
	return g.makeAsyncAssertion(AsyncAssertionTypeConsistently, 0, args...)
}

func (g *Gomega) ConsistentlyWithOffset(offset int, args ...interface{}) types.AsyncAssertion {
	return g.makeAsyncAssertion(AsyncAssertionTypeConsistently, offset, args...)
}

func (g *Gomega) makeAsyncAssertion(asyncAssertionType AsyncAssertionType, offset int, args ...interface{}) types.AsyncAssertion {
	baseOffset := 3
	timeoutInterval := -time.Duration(1)
	pollingInterval := -time.Duration(1)
	intervals := []interface{}{}
	var ctx context.Context
	if len(args) == 0 {
		g.Fail(fmt.Sprintf("Call to %s is missing a value or function to poll", asyncAssertionType), offset+baseOffset)
		return nil
	}

	actual := args[0]
	startingIndex := 1
	if _, isCtx := args[0].(context.Context); isCtx && len(args) > 1 {
		// the first argument is a context, we should accept it as the context _only if_ it is **not** the only argumnent **and** the second argument is not a parseable duration
		// this is due to an unfortunate ambiguity in early version of Gomega in which multi-type durations are allowed after the actual
		if _, err := toDuration(args[1]); err != nil {
			ctx = args[0].(context.Context)
			actual = args[1]
			startingIndex = 2
		}
	}

	for _, arg := range args[startingIndex:] {
		switch v := arg.(type) {
		case context.Context:
			ctx = v
		default:
			intervals = append(intervals, arg)
		}
	}
	var err error
	if len(intervals) > 0 {
		timeoutInterval, err = toDuration(intervals[0])
		if err != nil {
			g.Fail(err.Error(), offset+baseOffset)
		}
	}
	if len(intervals) > 1 {
		pollingInterval, err = toDuration(intervals[1])
		if err != nil {
			g.Fail(err.Error(), offset+baseOffset)
		}
	}

	return NewAsyncAssertion(asyncAssertionType, actual, g, timeoutInterval, pollingInterval, ctx, offset)
}

func (g *Gomega) SetDefaultEventuallyTimeout(t time.Duration) {
	g.DurationBundle.EventuallyTimeout = t
}

func (g *Gomega) SetDefaultEventuallyPollingInterval(t time.Duration) {
	g.DurationBundle.EventuallyPollingInterval = t
}

func (g *Gomega) SetDefaultConsistentlyDuration(t time.Duration) {
	g.DurationBundle.ConsistentlyDuration = t
}

func (g *Gomega) SetDefaultConsistentlyPollingInterval(t time.Duration) {
	g.DurationBundle.ConsistentlyPollingInterval = t
}
