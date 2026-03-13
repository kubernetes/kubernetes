// Copyright (c) 2017 Uber Technologies, Inc.
//
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
	"fmt"
	"io"
	"os"
)

// Variables for stubbing in unit tests.
var (
	_osExit             = os.Exit
	_osStderr io.Writer = os.Stderr
)

// TestingM is the minimal subset of testing.M that we use.
type TestingM interface {
	Run() int
}

// VerifyTestMain can be used in a TestMain function for package tests to
// verify that there were no goroutine leaks.
// To use it, your TestMain function should look like:
//
//	func TestMain(m *testing.M) {
//	  goleak.VerifyTestMain(m)
//	}
//
// See https://golang.org/pkg/testing/#hdr-Main for more details.
//
// This will run all tests as per normal, and if they were successful, look
// for any goroutine leaks and fail the tests if any leaks were found.
func VerifyTestMain(m TestingM, options ...Option) {
	exitCode := m.Run()
	opts := buildOpts(options...)

	var cleanup func(int)
	cleanup, opts.cleanup = opts.cleanup, nil
	if cleanup == nil {
		cleanup = _osExit
	}
	defer func() { cleanup(exitCode) }()

	if exitCode == 0 {
		if err := Find(opts); err != nil {
			fmt.Fprintf(_osStderr, "goleak: Errors on successful test run: %v\n", err)
			exitCode = 1
		}
	}
}
