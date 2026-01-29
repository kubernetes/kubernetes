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

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type SchedulerPerfOption func(options *schedulerPerfOptions)

// PrepareFn is a function that is called before each workload is run.
type PrepareFn func(tCtx ktesting.TContext) error

// PreInitFn is a function that is called before each workload's test case is set up.
type PreInitFn func(tb ktesting.TB, w *Workload) (func(), error)

// NodeUpdateFn is a function called after nodes are created in a workload.
type NodeUpdateFn func(tCtx ktesting.TContext, w *Workload, nodes *v1.NodeList) error

type schedulerPerfOptions struct {
	prepareFn    PrepareFn
	preInitFn    PreInitFn
	nodeUpdateFn NodeUpdateFn
}

// WithPrepareFn is the option to set a function that is called
// before each workload is run. (e.g. applying CRDs for custom plugins)
func WithPrepareFn(prepareFn PrepareFn) SchedulerPerfOption {
	return func(s *schedulerPerfOptions) {
		s.prepareFn = prepareFn
	}
}

// WithPreInitFn is the option to set a function that is called
// before the scheduler plugins are initialized for each workload's test case.
func WithPreInitFn(preInitFn PreInitFn) SchedulerPerfOption {
	return func(s *schedulerPerfOptions) {
		s.preInitFn = preInitFn
	}
}

// WithNodeUpdateFn is the option to set a function that is called
// after nodes are created within a workload execution.
func WithNodeUpdateFn(fn NodeUpdateFn) SchedulerPerfOption {
	return func(s *schedulerPerfOptions) {
		s.nodeUpdateFn = fn
	}
}
