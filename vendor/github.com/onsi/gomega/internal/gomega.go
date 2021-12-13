package internal

import (
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

func (g *Gomega) Eventually(actual interface{}, intervals ...interface{}) types.AsyncAssertion {
	return g.EventuallyWithOffset(0, actual, intervals...)
}

func (g *Gomega) EventuallyWithOffset(offset int, actual interface{}, intervals ...interface{}) types.AsyncAssertion {
	timeoutInterval := g.DurationBundle.EventuallyTimeout
	pollingInterval := g.DurationBundle.EventuallyPollingInterval
	if len(intervals) > 0 {
		timeoutInterval = toDuration(intervals[0])
	}
	if len(intervals) > 1 {
		pollingInterval = toDuration(intervals[1])
	}

	return NewAsyncAssertion(AsyncAssertionTypeEventually, actual, g, timeoutInterval, pollingInterval, offset)
}

func (g *Gomega) Consistently(actual interface{}, intervals ...interface{}) types.AsyncAssertion {
	return g.ConsistentlyWithOffset(0, actual, intervals...)
}

func (g *Gomega) ConsistentlyWithOffset(offset int, actual interface{}, intervals ...interface{}) types.AsyncAssertion {
	timeoutInterval := g.DurationBundle.ConsistentlyDuration
	pollingInterval := g.DurationBundle.ConsistentlyPollingInterval
	if len(intervals) > 0 {
		timeoutInterval = toDuration(intervals[0])
	}
	if len(intervals) > 1 {
		pollingInterval = toDuration(intervals[1])
	}

	return NewAsyncAssertion(AsyncAssertionTypeConsistently, actual, g, timeoutInterval, pollingInterval, offset)
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
