/*
Copyright 2024 The Kubernetes Authors.

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

package ktesting

import "strings"

// Deprecated: use tCtx.WithStep instead
func WithStep(tCtx TContext, step string) TContext {
	return tCtx.WithStep(step)
}

// WithStep creates a context where a prefix is added to all errors and log
// messages, similar to how errors are wrapped. This can be nested, leaving a
// trail of "bread crumbs" that help figure out where in a test some problem
// occurred or why some log output gets written:
//
//	ERROR: bake cake: set heat for baking: oven not found
//
// The string should describe the operation that is about to happen ("starting
// the controller", "list items") or what is being operated on ("HTTP server").
// Multiple different prefixes get concatenated with a colon.
func (tc *TC) WithStep(step string) *TC {
	tc = tc.clone()
	tc.steps += step + ": "
	return tc
}

// Step is useful when the context with the step information is
// used more than once:
//
//	ktesting.Step(tCtx, "step 1", func(tCtx ktesting.TContext) {
//	 tCtx.Log(...)
//	    if (... ) {
//	       tCtx.Failf(...)
//	    }
//	)}
//
// Inside the callback, the tCtx variable is the one where the step
// has been added. This avoids the need to introduce multiple different
// context variables and risk of using the wrong one.
func Step(tCtx TContext, what string, cb func(tCtx TContext)) {
	tCtx.Helper()
	cb(WithStep(tCtx, what))
}

// TODO: remove Begin+End when not needed anymore

// Deprecated
func Begin(tCtx TContext, what string) TContext {
	return WithStep(tCtx, what)
}

// Deprecated
func End(tc *TC) TContext {
	// This is a quick hack to keep End working. Will be removed.
	tc = tc.clone()
	index := strings.LastIndex(strings.TrimSuffix(tc.steps, ": "), ": ")
	if index > 0 {
		tc.steps = tc.steps[:index+2]
	} else {
		tc.steps = ""
	}
	return tc
}

// Value intercepts a search for the special "GINKGO_SPEC_CONTEXT" and
// wraps the underlying reporter so that the steps are visible in the report.
func (tc *TC) Value(key any) any {
	if tc.steps != "" {
		if s, ok := key.(string); ok && s == ginkgoSpecContextKey {
			if reporter, ok := tc.Context.Value(key).(ginkgoReporter); ok {
				return ginkgoReporter(&stepReporter{reporter: reporter, steps: tc.steps})
			}
		}
	}
	return tc.Context.Value(key)
}

type stepReporter struct {
	reporter ginkgoReporter
	steps    string
}

var _ ginkgoReporter = &stepReporter{}

func (s *stepReporter) AttachProgressReporter(reporter func() string) func() {
	return s.reporter.AttachProgressReporter(func() string {
		report := reporter()
		return s.steps + report
	})
}
