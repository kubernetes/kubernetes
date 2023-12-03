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
		TContext: tCtx,
		what:     what,
	}
	return WithLogger(sCtx, klog.LoggerWithName(sCtx.Logger(), what))
}

type stepContext struct {
	TContext
	what string
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

// Value intercepts a search for the special
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
