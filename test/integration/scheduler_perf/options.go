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
	"k8s.io/kubernetes/pkg/scheduler"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

type SchedulerPerfOption func(options *schedulerPerfOptions)

// HookFn is a function that is called while going through the test execution.
// The function may record test errors or abort the current test run through tCtx.
// Alternatively, it may also return a non-nil error.
type HookFn func(tCtx ktesting.TContext) error

// PreRunFn hook function is called for each workload after feature gates are set,
// but before he scheduler is started. It returns an optional cleanup function and an error.
type PreRunFn func(tCtx ktesting.TContext, w *Workload) (func(), error)

// NodeUpdateFn is a function called after nodes are created in a workload using createNodesOp.
type NodeUpdateFn func(tCtx ktesting.TContext, scheduler *scheduler.Scheduler, w *Workload, nodes *v1.NodeList) error

type schedulerPerfOptions struct {
	outOfTreePluginRegistry frameworkruntime.Registry
	preRunFn                PreRunFn
	prepareFn               HookFn
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

// WithNodeUpdateFn is the option to set a function that is called
// after nodes are created by createNodesOp within a workload execution.
func WithNodeUpdateFn(fn NodeUpdateFn) SchedulerPerfOption {
	return func(s *schedulerPerfOptions) {
		s.nodeUpdateFn = fn
	}
}

// WithPreRunFn is the option to set a function that is called
// after configuring the process (logging, feature gates) and
// before running any code (etcd, scheduler).
func WithPreRunFn(preRunFn PreRunFn) SchedulerPerfOption {
	return func(s *schedulerPerfOptions) {
		s.preRunFn = preRunFn
	}
}
