// Copyright (c) 2016 Uber Technologies, Inc.
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

// Package exit provides stubs so that unit tests can exercise code that calls
// os.Exit(1).
package exit

import "os"

var _exit = os.Exit

// With terminates the process by calling os.Exit(code). If the package is
// stubbed, it instead records a call in the testing spy.
func With(code int) {
	_exit(code)
}

// A StubbedExit is a testing fake for os.Exit.
type StubbedExit struct {
	Exited bool
	Code   int
	prev   func(code int)
}

// Stub substitutes a fake for the call to os.Exit(1).
func Stub() *StubbedExit {
	s := &StubbedExit{prev: _exit}
	_exit = s.exit
	return s
}

// WithStub runs the supplied function with Exit stubbed. It returns the stub
// used, so that users can test whether the process would have crashed.
func WithStub(f func()) *StubbedExit {
	s := Stub()
	defer s.Unstub()
	f()
	return s
}

// Unstub restores the previous exit function.
func (se *StubbedExit) Unstub() {
	_exit = se.prev
}

func (se *StubbedExit) exit(code int) {
	se.Exited = true
	se.Code = code
}
