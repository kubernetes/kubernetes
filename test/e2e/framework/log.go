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
	"context"
	"fmt"
	"regexp"
	"runtime/debug"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/formatter"
	"github.com/onsi/ginkgo/v2/types"
	"github.com/onsi/gomega"

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

// By is a replacement for ginkgo.By which better supports long-running
// operations: it will print reminders that the step is still running at
// regular intervals. The length of that interval is configurable and defaults
// to TestContext.SlowStepThreshold, which itself defaults to 10 seconds. In
// addition, it will print the total runtime at the end of the operation if it
// exceeded the threshold.
func By(text string, callback func(), threshold ...time.Duration) {
	// TODO (?): add this functionality to ginkgo itself, see
	// https://github.com/onsi/gomega/issues/574. As it stands now, this
	// code was copied from ginkgo.By and modified. What had to be dropped
	// was the check that the suite is running because there's no API for it:
	// https://github.com/onsi/ginkgo/blob/4fbf0425aaa1e6242b73800b71fca772ee9ba2dc/core_dsl.go#L497-L499
	value := struct {
		Text     string
		Duration time.Duration
	}{
		Text: text,
	}
	_, config := ginkgo.GinkgoConfiguration()
	formatter := formatter.NewWithNoColorBool(config.NoColor)
	t := time.Now()
	ginkgo.GinkgoWriter.Println(formatter.F("{{bold}}STEP:{{/}} %s {{gray}}%s{{/}}", text, t.Format(types.GINKGO_TIME_FORMAT)))
	if len(threshold) > 1 {
		Fail("By takes at most one duration parameter", 1)
	}
	if len(threshold) == 0 {
		threshold = []time.Duration{TestContext.SlowStepThreshold}
	}

	ticker := time.NewTicker(threshold[0])
	defer ticker.Stop()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				ginkgo.GinkgoWriter.Println(formatter.F("{{bold}}STEP still running:{{/}} %s {{gray}}%s{{/}}", text, time.Now().Format(types.GINKGO_TIME_FORMAT)))
			}
		}
	}()

	ginkgo.AddReportEntry("By Step", ginkgo.ReportEntryVisibilityNever, ginkgo.Offset(1), &value, t)
	callback()
	value.Duration = time.Since(t)
	if value.Duration >= threshold[0] {
		ginkgo.GinkgoWriter.Println(formatter.F("{{bold}}STEP duration:{{/}} %s {{gray}}%s{{/}}", text, value.Duration))
	}
}

// SucceedEventually is a replacement for
// gomega.Eventually(callback, intervals...).Should(gomega.Succeed(), extra...).
//
// In contrast to gomega, it prints intermediate failures at regular intervals
// to provide early feedback that a test is stuck. The length of that interval
// can be given as third parameter after the timeout and poll interval. It
// defaults to 3 * TestContext.SlowStepThreshold.
//
// To get timing information for a long running SucceedEventually, wrap it in
// By.
func SucceedEventually(callback func(g gomega.Gomega), extra []interface{}, intervals ...interface{}) {
	SucceedEventuallyWithOffset(1, callback, extra, intervals...)
}

// SucceedEventuallyWithOffset is the same as SucceedEventually except that it can skip
// additional stackframes. With offset = 0 the direct caller is reported, just as in
// SucceedEventually.
func SucceedEventuallyWithOffset(offset int, callback func(g gomega.Gomega), extra []interface{}, intervals ...interface{}) {
	lastReport := time.Now()
	if len(intervals) > 3 {
		Fail("SucceedEventually only takes at most three duration values.", 1)
	}
	reportInterval := 3 * TestContext.SlowStepThreshold
	if len(intervals) == 3 {
		switch interval := intervals[2].(type) {
		case string:
			duration, err := time.ParseDuration(interval)
			if err != nil {
				Fail(fmt.Sprintf("SucceedEventually called with invalid report interval: %v", err), 1)
			}
			reportInterval = duration
		case time.Duration:
			reportInterval = interval
		default:
			Fail(fmt.Sprintf("SucceedEventually called with invalid report interval, need string or time.Duration: %v", interval), 1)
		}
		intervals = intervals[0:2]
	}

	gomega.EventuallyWithOffset(offset+1, func() string {
		failures := ""
		g := gomega.NewGomega(func(message string, callerSkip ...int) {
			failures = message
		})
		callback(g)
		if failures != "" {
			now := time.Now()
			if now.Sub(lastReport) >= reportInterval {
				header := ""
				if len(extra) > 0 {
					if format, ok := extra[0].(string); ok {
						header = fmt.Sprintf(format+"\n", extra[1:]...)
					} else {
						header = fmt.Sprintf("unexpected extra parameters: %v\n", extra)
					}
				}
				Logf("%s%s", header, failures)
				lastReport = now
			}
		}
		return failures
	}, intervals...).Should(gomega.BeEmpty(), extra...)
}

var codeFilterRE = regexp.MustCompile(`/github.com/onsi/ginkgo/v2/`)

// PrunedStack is a wrapper around debug.Stack() that removes information
// about the current goroutine and optionally skips some of the initial stack entries.
// With skip == 0, the returned stack will start with the caller of PruneStack.
// From the remaining entries it automatically filters out useless ones like
// entries coming from Ginkgo.
//
// This is a modified copy of PruneStack in https://github.com/onsi/ginkgo/v2/blob/f90f37d87fa6b1dd9625e2b1e83c23ffae3de228/internal/codelocation/code_location.go#L25:
//   - simplified API and thus renamed (calls debug.Stack() instead of taking a parameter)
//   - source code filtering updated to be specific to Kubernetes
//   - optimized to use bytes and in-place slice filtering from
//     https://github.com/golang/go/wiki/SliceTricks#filter-in-place
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
