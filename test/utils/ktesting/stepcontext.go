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
func (tCtx TContext) WithStep(step string) TContext {
	tCtx.steps += step + ": "
	return tCtx
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
func (tCtx TContext) Step(step string, cb func(tCtx TContext)) {
	tCtx.Helper()
	cb(tCtx.WithStep(step))
}

// Value intercepts a search for the special "GINKGO_SPEC_CONTEXT" and
// wraps the underlying reporter so that the steps are visible in the report.
func (tCtx TContext) Value(key any) any {
	if tCtx.steps != "" {
		if s, ok := key.(string); ok && s == ginkgoSpecContextKey {
			if reporter, ok := tCtx.Context.Value(key).(ginkgoReporter); ok {
				return ginkgoReporter(&stepReporter{reporter: reporter, steps: tCtx.steps})
			}
		}
	}
	return tCtx.Context.Value(key)
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
