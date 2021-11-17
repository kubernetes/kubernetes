/*
Copyright 2019 The Kubernetes Authors.

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

package framework

import (
	"bytes"
	"fmt"
	"regexp"
	"runtime/debug"
	"time"

	"github.com/onsi/ginkgo"

	// TODO: Remove the following imports (ref: https://github.com/kubernetes/kubernetes/issues/81245)
	e2eginkgowrapper "k8s.io/kubernetes/test/e2e/framework/ginkgowrapper"
)

func nowStamp() string {
	return time.Now().Format(time.StampMilli)
}

func log(level string, format string, args ...interface{}) {
	fmt.Fprintf(ginkgo.GinkgoWriter, nowStamp()+": "+level+": "+format+"\n", args...)
}

// Logf logs the info.
func Logf(format string, args ...interface{}) {
	log("INFO", format, args...)
}

// Failf logs the fail info, including a stack trace starts at 2 levels above its caller
// (for example, for call chain f -> g -> Failf("foo", ...) error would be logged for "f").
func Failf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	skip := 2
	log("FAIL", "%s\n\nFull Stack Trace\n%s", msg, PrunedStack(skip))
	e2eginkgowrapper.Fail(nowStamp()+": "+msg, skip)
	panic("unreachable")
}

// Fail is a replacement for ginkgo.Fail which logs the problem as it occurs
// together with a stack trace and then calls ginkgowrapper.Fail.
func Fail(msg string, callerSkip ...int) {
	skip := 1
	if len(callerSkip) > 0 {
		skip += callerSkip[0]
	}
	log("FAIL", "%s\n\nFull Stack Trace\n%s", msg, PrunedStack(skip))
	e2eginkgowrapper.Fail(nowStamp()+": "+msg, skip)
}

var codeFilterRE = regexp.MustCompile(`/github.com/onsi/ginkgo/`)

// PrunedStack is a wrapper around debug.Stack() that removes information
// about the current goroutine and optionally skips some of the initial stack entries.
// With skip == 0, the returned stack will start with the caller of PruneStack.
// From the remaining entries it automatically filters out useless ones like
// entries coming from Ginkgo.
//
// This is a modified copy of PruneStack in https://github.com/onsi/ginkgo/blob/f90f37d87fa6b1dd9625e2b1e83c23ffae3de228/internal/codelocation/code_location.go#L25:
// - simplified API and thus renamed (calls debug.Stack() instead of taking a parameter)
// - source code filtering updated to be specific to Kubernetes
// - optimized to use bytes and in-place slice filtering from
//   https://github.com/golang/go/wiki/SliceTricks#filter-in-place
func PrunedStack(skip int) []byte {
	fullStackTrace := debug.Stack()
	stack := bytes.Split(fullStackTrace, []byte("\n"))
	// Ensure that the even entries are the method names and the
	// the odd entries the source code information.
	if len(stack) > 0 && bytes.HasPrefix(stack[0], []byte("goroutine ")) {
		// Ignore "goroutine 29 [running]:" line.
		stack = stack[1:]
	}
	// The "+2" is for skipping over:
	// - runtime/debug.Stack()
	// - PrunedStack()
	skip += 2
	if len(stack) > 2*skip {
		stack = stack[2*skip:]
	}
	n := 0
	for i := 0; i < len(stack)/2; i++ {
		// We filter out based on the source code file name.
		if !codeFilterRE.Match([]byte(stack[i*2+1])) {
			stack[n] = stack[i*2]
			stack[n+1] = stack[i*2+1]
			n += 2
		}
	}
	stack = stack[:n]

	return bytes.Join(stack, []byte("\n"))
}
