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
	"errors"
	"fmt"
	"strings"
)

// WithError creates a context where test failures are collected and stored in
// the provided error instance when the caller is done. Use it like this:
//
//	func doSomething(tCtx ktesting.TContext) (finalErr error) {
//	     tCtx, finalize := WithError(tCtx, &finalErr)
//	     defer finalize()
//	     ...
//	     tCtx.Fatal("some failure")
//
// Any error already stored in the variable will get overwritten by finalize if
// there were test failures, otherwise the variable is left unchanged.
// If there were multiple test errors, then the error will wrap all of
// them with errors.Join.
//
// Test failures are not propagated to the parent context.
// WithRESTConfig initializes all client-go clients with new clients
// created for the config. The current test name gets included in the UserAgent.
func (tc *TC) WithError(err *error) (TContext, func()) {
	tc = tc.clone()
	tc.capture = &capture{}

	return tc, func() {
		// Recover has to be called in the deferred function. When called inside
		// a function called by a deferred function (like finalize below), it
		// returns nil.
		if e := recover(); e != nil {
			if _, ok := e.(fatalWithError); !ok {
				// Not our own panic, pass it on instead of setting the error.
				panic(e)
			}
		}

		tc.finalize(err)
	}
}

func (tc *TC) finalize(err *error) {
	tc.capture.mutex.Lock()
	defer tc.capture.mutex.Unlock()

	errs := tc.capture.errors
	if tc.capture.failed && len(errs) == 0 {
		errs = []error{errFailedWithNoExplanation}
	}
	if len(errs) == 0 {
		return
	}
	*err = failures{errors.Join(errs...)}
}

type failures struct {
	error
}

func (e failures) GomegaString() string {
	// We don't need to repeat the string. Errors already get formatted once by Gomega itself,
	// then it calls GomegaString for a summary that isn't necessary anymore.
	return ""
}

// buildHeader handles:
// - "ERROR:<non-empty prefix><optional header><suffix>" -> use both prefix and suffix when we have a header, otherwise just the suffix
// - "<empty prefix><optional header><suffix>" -> use suffix only if we have a header
func (tc *TC) buildHeader(prefix, suffix string) string {
	if tc.perTestHeader != nil {
		return prefix + tc.perTestHeader() + suffix
	}
	if prefix != "" {
		return suffix
	}
	return ""
}

// indent either indents all follow-up lines or all lines including the first one.
func indent(msg string, all bool) string {
	header := ""
	if all {
		header = "\t"
	}
	return header + strings.ReplaceAll(msg, "\n", "\n\t")
}

func (tc *TC) Skip(args ...any) {
	tc.Helper()
	// Enable `go vet printf` by directly calling fmt.Sprintln.
	msg := strings.TrimSpace(fmt.Sprintln(args...))
	tc.TB().Skip("SKIP:", tc.buildHeader(" ", " ")+tc.steps+indent(msg, false))
}

func (tc *TC) Skipf(format string, args ...any) {
	tc.Helper()
	// Enable `go vet printf` by directly calling fmt.Sprintf.
	msg := strings.TrimSpace(fmt.Sprintf(format, args...))
	tc.TB().Skip("SKIP:", tc.buildHeader(" ", " ")+tc.steps+indent(msg, false))
}

func (tc *TC) Log(args ...any) {
	tc.Helper()
	// Enable `go vet printf` by directly calling fmt.Sprintln.
	msg := strings.TrimSpace(fmt.Sprintln(args...))
	tc.TB().Log(tc.buildHeader("", " ") + tc.steps + indent(msg, false))
}

func (tc *TC) Logf(format string, args ...any) {
	tc.Helper()
	// Enable `go vet printf` by directly calling fmt.Sprintf.
	msg := strings.TrimSpace(fmt.Sprintf(format, args...))
	tc.TB().Log(tc.buildHeader("", " ") + tc.steps + indent(msg, false))
}

func (tc *TC) Error(args ...any) {
	if tc.capture == nil {
		tc.Helper()
		msg := strings.TrimSpace(fmt.Sprintln(args...))
		// ERROR *before* header to make it stand out as failure.
		tc.TB().Error("ERROR:" + tc.buildHeader(" ", "\n") + indent(tc.steps+msg, true))
		return
	}

	tc.capture.mutex.Lock()
	defer tc.capture.mutex.Unlock()

	// Gomega adds a leading newline in https://github.com/onsi/gomega/blob/f804ac6ada8d36164ecae0513295de8affce1245/internal/gomega.go#L37
	// Let's strip that at start and end because ktesting will make errors
	// stand out more with the "ERROR" prefix, so there's no need for additional
	// line breaks. Besides, Sprintln (required for `go vet printf`) also
	// adds a trailing newline that we don't want.
	msg := strings.TrimSpace(fmt.Sprintln(args...))
	tc.capture.errors = append(tc.capture.errors, errors.New(tc.steps+msg))
	tc.capture.failed = true
}

func (tc *TC) Errorf(format string, args ...any) {
	if tc.capture == nil {
		tc.Helper()
		// Enable `go vet printf` by directly calling fmt.Sprintln.
		msg := strings.TrimSpace(fmt.Sprintf(format, args...))
		// ERROR *before* header to make it stand out as failure.
		tc.TB().Error("ERROR:" + tc.buildHeader(" ", "\n") + indent(tc.steps+msg, true))
		return
	}

	tc.capture.mutex.Lock()
	defer tc.capture.mutex.Unlock()

	msg := strings.TrimSpace(fmt.Sprintf(format, args...))
	tc.capture.errors = append(tc.capture.errors, errors.New(tc.steps+msg))
	tc.capture.failed = true
}

func (tc *TC) Fail() {
	if tc.capture == nil {
		tc.TB().Fail()
		return
	}

	tc.capture.mutex.Lock()
	defer tc.capture.mutex.Unlock()

	tc.capture.failed = true
}

func (tc *TC) FailNow() {
	if tc.capture == nil {
		tc.TB().FailNow()
		return
	}

	tc.capture.mutex.Lock()
	defer tc.capture.mutex.Unlock()

	tc.capture.failed = true
	panic(failed)
}

func (tc *TC) Failed() bool {
	if tc.capture == nil {
		return tc.TB().Failed()
	}

	tc.capture.mutex.Lock()
	defer tc.capture.mutex.Unlock()

	return tc.capture.failed
}

func (tc *TC) Fatal(args ...any) {
	if tc.capture == nil {
		tc.Helper()
		// Enable `go vet printf` by directly calling fmt.Sprintln.
		msg := strings.TrimSpace(fmt.Sprintln(args...))
		// FATAL ERROR *before* header to make it stand out as failure.
		tc.TB().Fatal("FATAL ERROR:" + tc.buildHeader(" ", "\n") + indent(tc.steps+msg, true))
	}

	tc.Error(args...)
	tc.FailNow()
}

func (tc *TC) Fatalf(format string, args ...any) {
	if tc.capture == nil {
		tc.Helper()
		// Enable `go vet printf` by directly calling fmt.Sprintf.
		msg := strings.TrimSpace(fmt.Sprintf(format, args...))
		// FATAL ERROR *before* header to make it stand out as failure.
		tc.TB().Fatal("FATAL ERROR:" + tc.buildHeader(" ", "\n") + indent(tc.steps+msg, true))
		return
	}

	tc.Errorf(format, args...)
	tc.FailNow()
}

// fatalWithError is the internal type that should never get propagated up. The
// only case where that can happen is when the developer forgot to call
// finalize via defer. The string explains that, in case that developers get to
// see it.
type fatalWithError string

const failed = fatalWithError("WithError TContext encountered a fatal error, but the finalize function was not called via defer as it should have been.")

var errFailedWithNoExplanation = errors.New("WithError context was marked as failed without recording an error")
