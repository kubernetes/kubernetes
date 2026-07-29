// Copyright (c) 2017 Uber Technologies, Inc.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package goleak

import (
	"errors"
	"fmt"

	"go.uber.org/goleak/internal/stack"
)

// TestingT is the minimal subset of testing.TB that we use.
type TestingT interface {
	Error(...interface{})
}

// filterStacks will filter any stacks excluded by the given opts.
// filterStacks modifies the passed in stacks slice.
func filterStacks(stacks []stack.Stack, skipID int, opts *opts) []stack.Stack {
	filtered := stacks[:0]
	for _, stack := range stacks {
		// Always skip the running goroutine.
		if stack.ID() == skipID {
			continue
		}
		// Run any default or user-specified filters.
		if opts.filter(stack) {
			continue
		}
		filtered = append(filtered, stack)
	}
	return filtered
}

// Find looks for extra goroutines, and returns a descriptive error if
// any are found.
func Find(options ...Option) error {
	cur := stack.Current().ID()

	opts := buildOpts(options...)
	if opts.cleanup != nil {
		return errors.New("Cleanup can only be passed to VerifyNone or VerifyTestMain")
	}
	var stacks []stack.Stack
	retry := true
	for i := 0; retry; i++ {
		stacks = filterStacks(stack.All(), cur, opts)

		if len(stacks) == 0 {
			return nil
		}
		retry = opts.retry(i)
	}

	return fmt.Errorf("found unexpected goroutines:\n%s", stacks)
}

type testHelper interface {
	Helper()
}

// VerifyNone marks the given TestingT as failed if any extra goroutines are
// found by Find. This is a helper method to make it easier to integrate in
// tests by doing:
//
//	defer VerifyNone(t)
//
// VerifyNone is currently incompatible with t.Parallel because it cannot
// associate specific goroutines with specific tests. Thus, non-leaking
// goroutines from other tests running in parallel could fail this check.
// If you need to run tests in parallel, use [VerifyTestMain] instead,
// which will verify that no leaking goroutines exist after ALL tests finish.
func VerifyNone(t TestingT, options ...Option) {
	opts := buildOpts(options...)
	var cleanup func(int)
	cleanup, opts.cleanup = opts.cleanup, nil

	if h, ok := t.(testHelper); ok {
		// Mark this function as a test helper, if available.
		h.Helper()
	}

	if err := Find(opts); err != nil {
		t.Error(err)
	}

	if cleanup != nil {
		cleanup(0)
	}
}
