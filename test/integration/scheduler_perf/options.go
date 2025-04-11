/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
