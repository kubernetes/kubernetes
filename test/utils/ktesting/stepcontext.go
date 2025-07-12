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

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/klog/v2"
)

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
func WithStep(tCtx TContext, what string) TContext {
	sCtx := &stepContext{
		TContext:  WithLogger(tCtx, klog.LoggerWithName(tCtx.Logger(), what)),
		parentCtx: tCtx,
		what:      what,
		start:     time.Now(),
	}
	return sCtx
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

// Begin and End can be used instead of Step to execute some instructions
// with a new context without using a callback method. This is useful
// when some local variables need to be set which are read later one.
// Log entries document the start and end of the step, including its duration.
//
//	tCtx = ktesting.Begin(tCtx, "step 1")
//	.. do something with tCtx
//	tCtx = ktesting.End(tCtx)
func Begin(tCtx TContext, what string) TContext {
	tCtx.Helper()
	tCtx = WithStep(tCtx, what)
	tCtx.Log("Starting...")
	return tCtx
}

// End complements Begin and returns the original context that was passed to Begin.
// It must be called on the context returned by Begin.
func End(tCtx TContext) TContext {
	tCtx.Helper()
	sCtx, ok := tCtx.(*stepContext)
	if !ok {
		tCtx.Fatalf("expected result of Begin, got instead %T", tCtx)
	}
	tCtx.Logf("Done, duration %s", time.Since(sCtx.start))
	return sCtx.parentCtx
}

type stepContext struct {
	TContext
	parentCtx TContext
	what      string
	start     time.Time
}

func (sCtx *stepContext) Log(args ...any) {
	sCtx.Helper()
	sCtx.TContext.Log(sCtx.what + ": " + strings.TrimSpace(fmt.Sprintln(args...)))
}

func (sCtx *stepContext) Logf(format string, args ...any) {
	sCtx.Helper()
	sCtx.TContext.Log(sCtx.what + ": " + strings.TrimSpace(fmt.Sprintf(format, args...)))
}

func (sCtx *stepContext) Error(args ...any) {
	sCtx.Helper()
	sCtx.TContext.Error(sCtx.what + ": " + strings.TrimSpace(fmt.Sprintln(args...)))
}

func (sCtx *stepContext) Errorf(format string, args ...any) {
	sCtx.Helper()
	sCtx.TContext.Error(sCtx.what + ": " + strings.TrimSpace(fmt.Sprintf(format, args...)))
}

func (sCtx *stepContext) Fatal(args ...any) {
	sCtx.Helper()
	sCtx.TContext.Fatal(sCtx.what + ": " + strings.TrimSpace(fmt.Sprintln(args...)))
}

func (sCtx *stepContext) Fatalf(format string, args ...any) {
	sCtx.Helper()
	sCtx.TContext.Fatal(sCtx.what + ": " + strings.TrimSpace(fmt.Sprintf(format, args...)))
}

// Value intercepts a search for the special "GINKGO_SPEC_CONTEXT".
func (sCtx *stepContext) Value(key any) any {
	if s, ok := key.(string); ok && s == ginkgoSpecContextKey {
		if reporter, ok := sCtx.TContext.Value(key).(ginkgoReporter); ok {
			return ginkgoReporter(&stepReporter{reporter: reporter, what: sCtx.what})
		}
	}
	return sCtx.TContext.Value(key)
}

type stepReporter struct {
	reporter ginkgoReporter
	what     string
}

var _ ginkgoReporter = &stepReporter{}

func (s *stepReporter) AttachProgressReporter(reporter func() string) func() {
	return s.reporter.AttachProgressReporter(func() string {
		report := reporter()
		return s.what + ": " + report
	})
}
