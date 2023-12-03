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
	"sync"

	"k8s.io/klog/v2"
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
func WithError(tCtx TContext, err *error) (TContext, func()) {
	eCtx := &errorContext{
		TContext: tCtx,
	}

	return eCtx, func() {
		// Recover has to be called in the deferred function. When called inside
		// a function called by a deferred function (like finalize below), it
		// returns nil.
		if e := recover(); e != nil {
			if _, ok := e.(fatalWithError); !ok {
				// Not our own panic, pass it on instead of setting the error.
				panic(e)
			}
		}

		eCtx.finalize(err)
	}
}

type errorContext struct {
	TContext

	mutex  sync.Mutex
	errors []error
	failed bool
}

func (eCtx *errorContext) finalize(err *error) {
	eCtx.mutex.Lock()
	defer eCtx.mutex.Unlock()

	if !eCtx.failed {
		return
	}

	errs := eCtx.errors
	if len(errs) == 0 {
		errs = []error{errFailedWithNoExplanation}
	}
	*err = errors.Join(errs...)
}

func (eCtx *errorContext) Error(args ...any) {
	eCtx.mutex.Lock()
	defer eCtx.mutex.Unlock()

	// Gomega adds a leading newline in https://github.com/onsi/gomega/blob/f804ac6ada8d36164ecae0513295de8affce1245/internal/gomega.go#L37
	// Let's strip that at start and end because ktesting will make errors
	// stand out more with the "ERROR" prefix, so there's no need for additional
	// line breaks.
	eCtx.errors = append(eCtx.errors, errors.New(strings.TrimSpace(fmt.Sprintln(args...))))
	eCtx.failed = true
}

func (eCtx *errorContext) Errorf(format string, args ...any) {
	eCtx.mutex.Lock()
	defer eCtx.mutex.Unlock()

	eCtx.errors = append(eCtx.errors, errors.New(strings.TrimSpace(fmt.Sprintf(format, args...))))
	eCtx.failed = true
}

func (eCtx *errorContext) Fail() {
	eCtx.mutex.Lock()
	defer eCtx.mutex.Unlock()

	eCtx.failed = true
}

func (eCtx *errorContext) FailNow() {
	eCtx.Helper()
	eCtx.Fail()
	panic(failed)
}

func (eCtx *errorContext) Failed() bool {
	eCtx.mutex.Lock()
	defer eCtx.mutex.Unlock()

	return eCtx.failed
}

func (eCtx *errorContext) Fatal(args ...any) {
	eCtx.Error(args...)
	eCtx.FailNow()
}

func (eCtx *errorContext) Fatalf(format string, args ...any) {
	eCtx.Errorf(format, args...)
	eCtx.FailNow()
}

func (eCtx *errorContext) CleanupCtx(cb func(TContext)) {
	eCtx.Helper()
	cleanupCtx(eCtx, cb)
}

func (eCtx *errorContext) Logger() klog.Logger {
	return klog.FromContext(eCtx)
}

// fatalWithError is the internal type that should never get propagated up. The
// only case where that can happen is when the developer forgot to call
// finalize via defer. The string explains that, in case that developers get to
// see it.
type fatalWithError string

const failed = fatalWithError("WithError TContext encountered a fatal error, but the finalize function was not called via defer as it should have been.")

var errFailedWithNoExplanation = errors.New("WithError context was marked as failed without recording an error")
