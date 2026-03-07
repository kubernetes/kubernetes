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
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type SchedulerPerfOption func(options *schedulerPerfOptions)

// HookFn is a function that is called while going through the test execution.
// The function may record test errors or abort the current test run through tCtx.
// Alternatively, it may also return a non-nil error.
type HookFn func(tCtx ktesting.TContext) error

// PreInitFn is a function that is called before each workload's test case is set up.
type PreInitFn func(tb ktesting.TB, w *Workload) (func(), error)

// NodeUpdateFn is a function called after nodes are created in a workload.
type NodeUpdateFn func(tCtx ktesting.TContext, w *Workload, nodes *v1.NodeList) error

type schedulerPerfOptions struct {
	outOfTreePluginRegistry frameworkruntime.Registry
	preRunFn                HookFn
	prepareFn               HookFn
	preInitFn               PreInitFn
	nodeUpdateFn            NodeUpdateFn
}

// WithPrepareFn is the option to set a function that is called
// before each workload is run. (e.g. applying CRDs for custom plugins)
// Scheduler and etcd are started at that point.
func WithPrepareFn(prepareFn HookFn) SchedulerPerfOption {
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

// WithPreRunFn is the option to set a function that is called
// after configuring the process (logging, feature gates) and
// before running any code (etcd, scheduler).
func WithPreRunFn(preRunFn HookFn) SchedulerPerfOption {
	return func(s *schedulerPerfOptions) {
		s.preRunFn = preRunFn
	}
}
