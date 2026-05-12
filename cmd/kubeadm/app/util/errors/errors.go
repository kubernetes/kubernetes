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

// Package errors contains a local implementation of error wrapping
// with stack traces similar to https://github.com/pkg/errors.
// Note that this is a written from scratch, much simpler implementation
// and not a fork of pkg/errors.
package errors

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
)

const (
	// defaultErrorExitCode defines the generic error code of 1.
	defaultErrorExitCode = 1
)

// errorWithStack is a basic Error/Unwrap interface implementor used for wrapping
// and stack traces.
type errorWithStack struct {
	msg   error
	cause error
	stack string
}

func (e errorWithStack) Error() string {
	return e.msg.Error()
}

func (e errorWithStack) Unwrap() error {
	return e.cause
}

// Wrap wraps an error and includes stack.
func Wrap(err error, message string) error {
	if err == nil {
		return nil
	}
	return &errorWithStack{
		msg:   fmt.Errorf("%s: %w", message, err),
		cause: err,
		stack: callStack(),
	}
}

// Wrapf is the same as Wrap but supports formatting and arguments.
func Wrapf(err error, format string, args ...any) error {
	if err == nil {
		return nil
	}
	return &errorWithStack{
		msg:   fmt.Errorf("%s: %w", fmt.Sprintf(format, args...), err),
		cause: err,
		stack: callStack(),
	}
}

// Unwrap unwraps an error using the stdlib method.
func Unwrap(err error) error {
	return errors.Unwrap(err)
}

// WithMessage includes formats an error with a message and includes stack trace.
func WithMessage(err error, message string) error {
	if err == nil {
		return nil
	}
	return &errorWithStack{
		msg:   fmt.Errorf("%s: %s", message, err.Error()),
		stack: callStack(),
	}
}

// WithMessagef is the same as WithMessage but supports formatting and arguments.
func WithMessagef(err error, format string, args ...any) error {
	if err == nil {
		return nil
	}
	return &errorWithStack{
		msg:   fmt.Errorf("%s: %s", fmt.Sprintf(format, args...), err.Error()),
		stack: callStack(),
	}
}

// New calls the stdlib errors.New and includes a stack trace.
func New(message string) error {
	return &errorWithStack{
		msg:   errors.New(message),
		stack: callStack(),
	}
}

// Errorf is calls the stdlib errors.Errorf and includes a stack trace.
func Errorf(format string, args ...any) error {
	return &errorWithStack{
		msg:   fmt.Errorf(format, args...),
		stack: callStack(),
	}
}

// StackTrace retrieves the stack trace of an error as a string.
func StackTrace(err error) string {
	s, ok := err.(*errorWithStack) //nolint:errorlint
	if !ok {
		return "no stack trace"
	}
	return s.stack
}

// Is wraps the stdlib errors.Is.
func Is(err, target error) bool {
	return errors.Is(err, target)
}

// As wraps the stdlib errors.As.
func As(err error, target any) bool {
	return errors.As(err, target)
}

// callStack gets the call stack at the location that created an error.
// It skips 3 callers so that the location that created the error is last on top.
// Follows a similar formatting to runtime.PrintStack() and panic().
func callStack() string {
	pc := make([]uintptr, 32)
	n := runtime.Callers(3, pc)
	frames := runtime.CallersFrames(pc[:n])
	buf := bytes.Buffer{}

	for {
		frame, more := frames.Next()
		fmt.Fprintf(&buf, "%s\n\t%s:%d\n", frame.Function, frame.File, frame.Line)
		if !more {
			break
		}
	}

	return buf.String()
}

// HandleError is the default wrapper around handleError for handling errors
// that calls the exitWithCode function as a callback.
func HandleError(err error) {
	handleError(err, exitWithCode)
}

func handleError(err error, handleFunc func(string, int)) {
	if err == nil {
		return
	}

	msg := fmt.Sprintf("%s\nTo see the stack trace of this error execute with --v=5 or higher", err.Error())
	// Check if the verbosity level in klog is high enough and print a stack trace.
	f := flag.CommandLine.Lookup("v")
	if f != nil {
		// Assume that the "v" flag contains a parsable Int32 as per klog's "Level" type alias,
		// thus no error from ParseInt is handled here.
		if v, e := strconv.ParseInt(f.Value.String(), 10, 32); e == nil {
			// https://git.k8s.io/community/contributors/devel/sig-instrumentation/logging.md
			// klog.V(5) - Trace level verbosity
			if v > 4 {
				msg = fmt.Sprintf("%v\n%s", err.Error(), StackTrace(err))
			}
		}
	}

	handleFunc(msg, defaultErrorExitCode)
}

// exitWithCode optionally prints a message to stderr and then exits
// with the given error code.
func exitWithCode(msg string, code int) {
	if len(msg) > 0 {
		// add newline if needed
		if !strings.HasSuffix(msg, "\n") {
			msg += "\n"
		}

		_, _ = fmt.Fprintf(os.Stderr, "error: %s", msg)
	}
	os.Exit(code)
}
