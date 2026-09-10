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

package initoption

import (
	"time"

	"k8s.io/kubernetes/test/utils/ktesting/internal"
)

// InitOption is a functional option for Init and InitCtx.
type InitOption func(c *internal.InitConfig)

// PerTestOutput controls whether a per-test logger gets
// set up by Init. Has no effect in InitCtx.
//
// The default, used when this option is not passed to Init at all, is
// enabled (true).
//
// Example for a benchmark that wants production code to log through the
// normal, global klog configuration instead of through a per-test logger:
//
//	func BenchmarkSomething(b *testing.B) {
//	    tCtx := ktesting.Init(b, initoption.PerTestOutput(false))
//	    // ...
//	}
func PerTestOutput(enabled bool) InitOption {
	return func(c *internal.InitConfig) {
		c.PerTestOutput = enabled
	}
}

// BufferLogs controls whether log entries are captured in memory in addition
// to being printed. Off by default. Unit tests that want to verify that
// log entries are emitted as expected can turn this on and then retrieve
// the captured log through the Underlier LogSink interface:
//
//	func TestSomething(t *testing.T) {
//	    tCtx := ktesting.Init(t, initoption.BufferLogs(true))
//	    codeUnderTest(tCtx)
//	    underlier, ok := tCtx.Logger().GetSink().(ktesting.Underlier)
//	    if !ok {
//	        t.Fatal("should have had an ktesting log sink")
//	    }
//	    log := underlier.GetBuffer().Data()
//	    // Check log entries...
//	}
func BufferLogs(enabled bool) InitOption {
	return func(c *internal.InitConfig) {
		c.BufferLogs = enabled
	}
}

// WithCleanupGracePeriod overrides the default cleanup grace period.
// The cleanup grace period is the time reserved before the test-suite
// deadline so that cleanup callbacks can complete before the test binary is
// killed.
//
// A non-positive value is ignored and the default is used instead.
//
// When using Ginkgo to manage the test suite this option has no effect because
// Ginkgo itself manages timeouts.
func WithCleanupGracePeriod(d time.Duration) InitOption {
	return func(c *internal.InitConfig) {
		c.CleanupGracePeriod = d
	}
}
