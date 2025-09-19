package internal

import (
	"context"
	"reflect"

	"github.com/onsi/ginkgo/v2/types"
)

type SpecContext interface {
	context.Context

	SpecReport() types.SpecReport
	AttachProgressReporter(func() string) func()
	WrappedContext() context.Context
}

type specContext struct {
	context.Context
	*ProgressReporterManager

	cancel context.CancelCauseFunc

	suite *Suite
}

/*
SpecContext includes a reference to `suite` and embeds itself in itself as a "GINKGO_SPEC_CONTEXT" value.  This allows users to create child Contexts without having down-stream consumers (e.g. Gomega) lose access to the SpecContext and its methods.  This allows us to build extensions on top of Ginkgo that simply take an all-encompassing context.

Note that while SpecContext is used to enforce deadlines by Ginkgo it is not configured as a context.WithDeadline.  Instead, Ginkgo owns responsibility for cancelling the context when the deadline elapses.

This is because Ginkgo needs finer control over when the context is canceled.  Specifically, Ginkgo needs to generate a ProgressReport before it cancels the context to ensure progress is captured where the spec is currently running.  The only way to avoid a race here is to manually control the cancellation.
*/
func NewSpecContext(suite *Suite) *specContext {
	ctx, cancel := context.WithCancelCause(context.Background())
	sc := &specContext{
		cancel:                  cancel,
		suite:                   suite,
		ProgressReporterManager: NewProgressReporterManager(),
	}
	ctx = context.WithValue(ctx, "GINKGO_SPEC_CONTEXT", sc) //yes, yes, the go docs say don't use a string for a key... but we'd rather avoid a circular dependency between Gomega and Ginkgo
	sc.Context = ctx                                        //thank goodness for garbage collectors that can handle circular dependencies

	return sc
}

func (sc *specContext) SpecReport() types.SpecReport {
	return sc.suite.CurrentSpecReport()
}

func (sc *specContext) WrappedContext() context.Context {
	return sc.Context
}

/*
The user is allowed to wrap `SpecContext` in a new context.Context when using AroundNodes.  But body functions expect SpecContext.
We support this by taking their context.Context and returning a SpecContext that wraps it.
*/
func wrapContextChain(ctx context.Context) SpecContext {
	if ctx == nil {
		return nil
	}
	if reflect.TypeOf(ctx) == reflect.TypeOf(&specContext{}) {
		return ctx.(*specContext)
	} else if sc, ok := ctx.Value("GINKGO_SPEC_CONTEXT").(*specContext); ok {
		return &specContext{
			Context:                 ctx,
			ProgressReporterManager: sc.ProgressReporterManager,
			cancel:                  sc.cancel,
			suite:                   sc.suite,
		}
	}
	return nil
}
