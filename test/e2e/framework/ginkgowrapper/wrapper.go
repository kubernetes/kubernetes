/*
Copyright 2017 The Kubernetes Authors.

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

// Package ginkgowrapper wraps Ginkgo Fail and Skip functions to panic
// with structured data instead of a constant string.
package ginkgowrapper

import (
	"bufio"
	"bytes"
	"regexp"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/onsi/ginkgo"
)

// FailurePanic is the value that will be panicked from Fail.
type FailurePanic struct {
	Message        string // The failure message passed to Fail
	Filename       string // The filename that is the source of the failure
	Line           int    // The line number of the filename that is the source of the failure
	FullStackTrace string // A full stack trace starting at the source of the failure
}

// String makes FailurePanic look like the old Ginkgo panic when printed.
func (FailurePanic) String() string { return ginkgo.GINKGO_PANIC }

// Fail wraps ginkgo.Fail so that it panics with more useful
// information about the failure. This function will panic with a
// FailurePanic.
func Fail(message string, callerSkip ...int) {
	skip := 1
	if len(callerSkip) > 0 {
		skip += callerSkip[0]
	}

	_, file, line, _ := runtime.Caller(skip)
	fp := FailurePanic{
		Message:        message,
		Filename:       file,
		Line:           line,
		FullStackTrace: pruneStack(skip),
	}

	defer func() {
		e := recover()
		if e != nil {
			panic(fp)
		}
	}()

	ginkgo.Fail(message, skip)
}

// SkipPanic is the value that will be panicked from Skip.
type SkipPanic struct {
	Message        string // The failure message passed to Fail
	Filename       string // The filename that is the source of the failure
	Line           int    // The line number of the filename that is the source of the failure
	FullStackTrace string // A full stack trace starting at the source of the failure
}

// String makes SkipPanic look like the old Ginkgo panic when printed.
func (SkipPanic) String() string { return ginkgo.GINKGO_PANIC }

// Skip wraps ginkgo.Skip so that it panics with more useful
// information about why the test is being skipped. This function will
// panic with a SkipPanic.
func Skip(message string, callerSkip ...int) {
	skip := 1
	if len(callerSkip) > 0 {
		skip += callerSkip[0]
	}

	_, file, line, _ := runtime.Caller(skip)
	sp := SkipPanic{
		Message:        message,
		Filename:       file,
		Line:           line,
		FullStackTrace: pruneStack(skip),
	}

	defer func() {
		e := recover()
		if e != nil {
			panic(sp)
		}
	}()

	ginkgo.Skip(message, skip)
}

// ginkgo adds a lot of test running infrastructure to the stack, so
// we filter those out
var stackSkipPattern = regexp.MustCompile(`onsi/ginkgo`)

func pruneStack(skip int) string {
	skip += 2 // one for pruneStack and one for debug.Stack
	stack := debug.Stack()
	scanner := bufio.NewScanner(bytes.NewBuffer(stack))
	var prunedStack []string

	// skip the top of the stack
	for i := 0; i < 2*skip+1; i++ {
		scanner.Scan()
	}

	for scanner.Scan() {
		if stackSkipPattern.Match(scanner.Bytes()) {
			scanner.Scan() // these come in pairs
		} else {
			prunedStack = append(prunedStack, scanner.Text())
			scanner.Scan() // these come in pairs
			prunedStack = append(prunedStack, scanner.Text())
		}
	}

	return strings.Join(prunedStack, "\n")
}
