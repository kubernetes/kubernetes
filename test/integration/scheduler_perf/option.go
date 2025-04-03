package benchmark

import "k8s.io/kubernetes/test/utils/ktesting"

type SchedulerPerfOption func(options *schedulerPerfOptions)

// PrepareFn is a function that is called before the benchmarks run.
type PrepareFn func(tCtx ktesting.TContext) error

type schedulerPerfOptions struct {
	prepareFn PrepareFn
}

// WithPrepareFn is the option to set a function that is called
// before the benchmarks run. (e.g. applying CRDs for custom plugins)
func WithPrepareFn(prepareFn PrepareFn) SchedulerPerfOption {
	return func(s *schedulerPerfOptions) {
		s.prepareFn = prepareFn
	}
}
