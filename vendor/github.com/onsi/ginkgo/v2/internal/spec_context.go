package internal

import (
	"context"
	"sort"
	"sync"

	"github.com/onsi/ginkgo/v2/types"
)

type SpecContext interface {
	context.Context

	SpecReport() types.SpecReport
	AttachProgressReporter(func() string) func()
}

type specContext struct {
	context.Context

	cancel            context.CancelFunc
	lock              *sync.Mutex
	progressReporters map[int]func() string
	prCounter         int

	suite *Suite
}

/*
SpecContext includes a reference to `suite` and embeds itself in itself as a "GINKGO_SPEC_CONTEXT" value.  This allows users to create child Contexts without having down-stream consumers (e.g. Gomega) lose access to the SpecContext and its methods.  This allows us to build extensions on top of Ginkgo that simply take an all-encompassing context.

Note that while SpecContext is used to enforce deadlines by Ginkgo it is not configured as a context.WithDeadline.  Instead, Ginkgo owns responsibility for cancelling the context when the deadline elapses.

This is because Ginkgo needs finer control over when the context is canceled.  Specifically, Ginkgo needs to generate a ProgressReport before it cancels the context to ensure progress is captured where the spec is currently running.  The only way to avoid a race here is to manually control the cancellation.
*/
func NewSpecContext(suite *Suite) *specContext {
	ctx, cancel := context.WithCancel(context.Background())
	sc := &specContext{
		cancel:            cancel,
		suite:             suite,
		lock:              &sync.Mutex{},
		prCounter:         0,
		progressReporters: map[int]func() string{},
	}
	ctx = context.WithValue(ctx, "GINKGO_SPEC_CONTEXT", sc) //yes, yes, the go docs say don't use a string for a key... but we'd rather avoid a circular dependency between Gomega and Ginkgo
	sc.Context = ctx                                        //thank goodness for garbage collectors that can handle circular dependencies

	return sc
}

func (sc *specContext) SpecReport() types.SpecReport {
	return sc.suite.CurrentSpecReport()
}

func (sc *specContext) AttachProgressReporter(reporter func() string) func() {
	sc.lock.Lock()
	defer sc.lock.Unlock()
	sc.prCounter += 1
	prCounter := sc.prCounter
	sc.progressReporters[prCounter] = reporter

	return func() {
		sc.lock.Lock()
		defer sc.lock.Unlock()
		delete(sc.progressReporters, prCounter)
	}
}

func (sc *specContext) QueryProgressReporters() []string {
	sc.lock.Lock()
	keys := []int{}
	for key := range sc.progressReporters {
		keys = append(keys, key)
	}
	sort.Ints(keys)
	reporters := []func() string{}
	for _, key := range keys {
		reporters = append(reporters, sc.progressReporters[key])
	}
	sc.lock.Unlock()

	if len(reporters) == 0 {
		return nil
	}
	out := []string{}
	for _, reporter := range reporters {
		out = append(out, reporter())
	}
	return out
}
