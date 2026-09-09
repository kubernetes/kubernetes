/*
Copyright The Kubernetes Authors.

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

// This file is in package ktesting (not ktesting_test) so that it can access
// unexported fields of TContext.
package ktesting

import (
	"context"
	"errors"
	"io"
	"maps"
	"os"
	"slices"
	"testing"
	"testing/synctest"
	"time"

	"github.com/onsi/gomega"
	"go.uber.org/goleak"

	"k8s.io/kubernetes/test/utils/ktesting/initoption"
)

func TestSyncTestInit(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		// This must work inside a synctest bubble, despite Deadline panicking there.
		// We then don't have a deadline.
		tCtx := Init(t)
		deadline, ok := tCtx.Deadline()
		if ok {
			tCtx.Errorf("Expected no deadline, got %s", deadline)
		}
		if !tCtx.IsSyncTest() {
			tCtx.Errorf("Expected to run as synctest")
		}
		tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
	})
}

func TestNormalInit(t *testing.T) {
	// The outcome depends on how the unit test was started.
	// See below for deterministic deadline/no deadline testing.
	expectDeadline, expectOK := t.Deadline()
	expectDeadline = expectDeadline.Add(-DefaultCleanupGracePeriod)
	tCtx := Init(t)
	actualDeadline, actualOK := tCtx.Deadline()
	tCtx.Expect(actualOK).To(gomega.Equal(expectOK), "have deadline")
	if expectOK {
		tCtx.Expect(actualDeadline).To(gomega.BeTemporally("~", expectDeadline, 2*time.Second), "deadline")
	}
	if tCtx.IsSyncTest() {
		tCtx.Errorf("Expected to not run as synctest")
	}
	tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
}

// deadlineT2 mirrors the deadlineT helper in ktesting_test.
type deadlineT2 struct {
	TB
	deadline *time.Time
}

func (t *deadlineT2) Deadline() (time.Time, bool) {
	if t.deadline == nil {
		return time.Time{}, false
	}
	return *t.deadline, true
}

// TestDefaultCleanupGracePeriod verifies that cleanupGracePeriod is set to
// DefaultCleanupGracePeriod when no override is given.
func TestDefaultCleanupGracePeriod(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		mockDeadline := time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)
		mockT := &deadlineT2{TB: t, deadline: &mockDeadline}
		tCtx := Init(mockT)
		if tCtx.cleanupGracePeriod != DefaultCleanupGracePeriod {
			t.Errorf("expected cleanupGracePeriod %v, got %v", DefaultCleanupGracePeriod, tCtx.cleanupGracePeriod)
		}
		tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
	})
}

// TestCustomCleanupGracePeriod verifies that WithCleanupGracePeriod stores the
// override in cleanupGracePeriod and shifts the effective deadline accordingly.
func TestCustomCleanupGracePeriod(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		mockDeadline := time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)
		mockT := &deadlineT2{TB: t, deadline: &mockDeadline}
		custom := 30 * time.Second
		tCtx := Init(mockT, initoption.WithCleanupGracePeriod(custom))

		if tCtx.cleanupGracePeriod != custom {
			t.Errorf("expected cleanupGracePeriod %v, got %v", custom, tCtx.cleanupGracePeriod)
		}

		actualDeadline, ok := tCtx.Deadline()
		if !ok {
			t.Fatal("expected a deadline but got none")
		}
		expect := mockDeadline.Add(-custom)
		tCtx.Expect(actualDeadline).To(
			gomega.BeTemporally("==", expect),
			"context deadline should be shifted by the custom grace period")
		tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
	})
}

// TestInitCtxCleanupGracePeriod verifies that InitCtx applies
// WithCleanupGracePeriod via the previously ignored opts parameter.
func TestInitCtxCleanupGracePeriod(t *testing.T) {
	custom := 42 * time.Second
	tCtx := InitCtx(context.Background(), t, initoption.WithCleanupGracePeriod(custom))
	if tCtx.cleanupGracePeriod != custom {
		t.Errorf("expected cleanupGracePeriod %v, got %v", custom, tCtx.cleanupGracePeriod)
	}
	tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
}

// TestInitCtxDefaultCleanupGracePeriod verifies that InitCtx falls back to
// DefaultCleanupGracePeriod when no option is given.
func TestInitCtxDefaultCleanupGracePeriod(t *testing.T) {
	tCtx := InitCtx(context.Background(), t)
	if tCtx.cleanupGracePeriod != DefaultCleanupGracePeriod {
		t.Errorf("expected cleanupGracePeriod %v, got %v", DefaultCleanupGracePeriod, tCtx.cleanupGracePeriod)
	}
	tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
}

// getRunningTests reports all currently running tests, sorted by name.
func getRunningTests() []string {
	defaultProgressReporter.reportMutex.Lock()
	defer defaultProgressReporter.reportMutex.Unlock()

	return slices.Sorted(maps.Keys(defaultProgressReporter.runningTests))
}

func TestWithError(t *testing.T) {
	t.Run("panic", func(t *testing.T) {
		tCtx := Init(t)
		tCtx.Expect(func() {
			tCtx := Init(t)
			var err error
			_, finalize := tCtx.WithError(&err)
			defer finalize()

			panic("pass me through")
		}).To(gomega.Panic())
	})

	normalErr := errors.New("normal error")

	for name, tc := range map[string]struct {
		cb           func(TContext)
		expectNoFail bool
		expectError  string
	}{
		"none": {
			cb:           func(tCtx TContext) {},
			expectNoFail: true,
			expectError:  normalErr.Error(),
		},
		"Error": {
			cb: func(tCtx TContext) {
				tCtx.Error("some error")
			},
			expectError: "some error",
		},
		"Errorf": {
			cb: func(tCtx TContext) {
				tCtx.Errorf("some %s", "error")
			},
			expectError: "some error",
		},
		"Fatal": {
			cb: func(tCtx TContext) {
				tCtx.Fatal("some error")
				tCtx.Error("another error")
			},
			expectError: "some error",
		},
		"Fatalf": {
			cb: func(tCtx TContext) {
				tCtx.Fatalf("some %s", "error")
				tCtx.Error("another error")
			},
			expectError: "some error",
		},
		"Fail": {
			cb: func(tCtx TContext) {
				tCtx.Fatalf("some %s", "error")
				tCtx.Error("another error")
			},
			expectError: "some error",
		},
		"FailNow": {
			cb: func(tCtx TContext) {
				tCtx.FailNow()
				tCtx.Error("another error")
			},
			expectError: errFailedWithNoExplanation.Error(),
		},
		"many": {
			cb: func(tCtx TContext) {
				tCtx.Error("first error")
				tCtx.Error("second error")
			},
			expectError: `first error
second error`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			tCtx := Init(t)
			err := normalErr
			tCtx, finalize := tCtx.WithError(&err)
			func() {
				defer finalize()
				tc.cb(tCtx)
			}()

			if tc.expectNoFail {
				tCtx.Assert(tCtx.Failed()).To(gomega.BeFalseBecause("should have failed"))
			} else {
				tCtx.Assert(tCtx.Failed()).To(gomega.BeTrueBecause("should not have failed"))
			}
			if tc.expectError == "" {
				tCtx.Assert(err).To(gomega.Succeed())
			} else {
				tCtx.Assert(err).To(gomega.MatchError(gomega.Equal(tc.expectError)))
			}
		})
	}
}

func TestStepContext(t *testing.T) {
	for name, tc := range map[string]testcase{
		"output": {
			cb: func(tCtx TContext) {
				tCtx = tCtx.WithStep("step")
				tCtx.Log("Log", "a", "b", 42)
				tCtx.Logf("Logf %s %s %d", "a", "b", 42)
				tCtx.Error("Error", "a", "b", 42)
				tCtx.Errorf("Errorf %s %s %d", "a", "b", 42)
			},
			expectTrace: `(LOG) <klog header>: step: Log a b 42
(LOG) <klog header>: step: Logf a b 42
(ERROR) ERROR: <klog header>:
	step: Error a b 42
(ERROR) ERROR: <klog header>:
	step: Errorf a b 42
`,
		},
		"nested steps": {
			cb: func(tCtx TContext) {
				tCtx = tCtx.WithStep("step 1").WithStep("step 2")
				tCtx.Log("Log")
				tCtx.Error("Error")
				tCtx.Logger().Info("Info")
			},
			// Multiple steps get concatenated with "/", the same
			// separator klog uses for logger names, both for the
			// plain text prefix and for the logger's name.
			expectTrace: `(LOG) <klog header>: step 1/step 2: Log
(ERROR) ERROR: <klog header>:
	step 1/step 2: Error
(LOG) <klog header> step 1/step 2: Info
`,
		},
		"fatal": {
			cb: func(tCtx TContext) {
				tCtx = tCtx.WithStep("step")
				tCtx.Fatal("Error", "a", "b", 42)
				// not reached
				tCtx.Log("Log")
			},
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	step: Error a b 42
`,
		},
		"fatalf": {
			cb: func(tCtx TContext) {
				tCtx = tCtx.WithStep("step")
				tCtx.Fatalf("Error %s %s %d", "a", "b", 42)
				// not reached
				tCtx.Log("Log")
			},
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	step: Error a b 42
`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			tc.run(t)
		})
	}
}

func TestProgressReport(t *testing.T) {
	oldOut := defaultProgressReporter.out
	out := newOutputStream()
	defaultProgressReporter.out = out
	t.Cleanup(func() {
		goleak.VerifyNone(t)
		defaultProgressReporter.out = oldOut

		// If we get here, the defaultProgressReporter is not active anymore,
		// but the interrupt context should still be canceled.
		gomega.NewGomegaWithT(t).Expect(defaultProgressReporter.usageCount).To(gomega.Equal(int64(0)), "usage count")
		gomega.NewGomegaWithT(t).Expect(context.Cause(interruptCtx)).To(gomega.MatchError(gomega.Equal("received interrupt signal")), "interrupted persistently")

		// Reset for next test.
		interruptCtx, interrupted = context.WithCancelCause(context.Background())
	})

	// This must use a real testing.T, otherwise Init doesn't initialize signal handling.
	tCtx := Init(t)
	tCtx = tCtx.WithStep("step")
	removeReporter := tCtx.Value("GINKGO_SPEC_CONTEXT").(ginkgoReporter).AttachProgressReporter(func() string { return "hello world" })
	defer removeReporter()
	tCtx.Expect(tCtx.Value("some other key")).To(gomega.BeNil(), "value for unknown context value key")

	// Trigger report and wait for it.
	defaultProgressReporter.progressChannel <- os.Interrupt // Should be SIGUSR1, but that is not defined and it doesn't matter.
	report := <-out.stream
	tCtx.Expect(report).To(gomega.Equal(`You requested a progress report.
Currently running:
	TestProgressReport

TestProgressReport:
	step: hello world
`), "report")

	gomega.NewGomegaWithT(t).Expect(context.Cause(interruptCtx)).To(gomega.Succeed(), "not interrupted yet")
	defaultProgressReporter.signalChannel <- os.Interrupt
	message := <-out.stream
	tCtx.Expect(message).To(gomega.Equal(`

INFO: canceling test context: received interrupt signal

`))
	gomega.NewGomegaWithT(t).Eventually(func() error { return context.Cause(tCtx) }).WithTimeout(30*time.Second).To(gomega.MatchError(gomega.Equal("received interrupt signal")), "interrupted")
}

func TestProgressReportSubTest(t *testing.T) {
	oldOut := defaultProgressReporter.out
	out := newOutputStream()
	defaultProgressReporter.out = out
	t.Cleanup(func() {
		goleak.VerifyNone(t)
		defaultProgressReporter.out = oldOut

		// If we get here, the defaultProgressReporter is not active anymore,
		// but the interrupt context should still be canceled.
		gomega.NewGomegaWithT(t).Expect(defaultProgressReporter.usageCount).To(gomega.Equal(int64(0)), "usage count")
		gomega.NewGomegaWithT(t).Expect(context.Cause(interruptCtx)).To(gomega.MatchError(gomega.Equal("received interrupt signal")), "interrupted persistently")

		// Reset for next test.
		interruptCtx, interrupted = context.WithCancelCause(context.Background())
	})

	// This must use a real testing.T, otherwise Init doesn't initialize signal handling.
	tCtx := Init(t)

	// Sub-tests must show up in the "Currently running" list, in
	// addition to the parent test, and their progress report must be
	// indented and prefixed with their own (sub-test) name, with every
	// line of a multi-line report indented.
	tCtx.Run("sub", func(tCtx TContext) {
		removeReporter := tCtx.Value("GINKGO_SPEC_CONTEXT").(ginkgoReporter).AttachProgressReporter(func() string { return "line one\nline two" })
		defer removeReporter()

		defaultProgressReporter.progressChannel <- os.Interrupt // Should be SIGUSR1, but that is not defined and it doesn't matter.
		report := <-out.stream
		tCtx.Expect(report).To(gomega.Equal(`You requested a progress report.
Currently running:
	TestProgressReportSubTest
	TestProgressReportSubTest/sub

TestProgressReportSubTest/sub:
	line one
	line two
`), "report")
	})

	// After the sub-test finished, only the parent test remains.
	defaultProgressReporter.progressChannel <- os.Interrupt // Should be SIGUSR1, but that is not defined and it doesn't matter.
	report := <-out.stream
	tCtx.Expect(report).To(gomega.Equal(`You requested a progress report.
Currently running:
	TestProgressReportSubTest
Currently there is no information about test progress available.
`), "report after sub-test completion")

	gomega.NewGomegaWithT(t).Expect(context.Cause(interruptCtx)).To(gomega.Succeed(), "not interrupted yet")
	defaultProgressReporter.signalChannel <- os.Interrupt
	message := <-out.stream
	tCtx.Expect(message).To(gomega.Equal(`

INFO: canceling test context: received interrupt signal

`))
	gomega.NewGomegaWithT(t).Eventually(func() error { return context.Cause(tCtx) }).WithTimeout(30*time.Second).To(gomega.MatchError(gomega.Equal("received interrupt signal")), "interrupted")
}

// outputStream forwards exactly one Write call to a stream.
// A second Write call is an error and will panic.
type outputStream struct {
	stream chan string
}

var _ io.Writer = &outputStream{}

func newOutputStream() *outputStream {
	return &outputStream{
		stream: make(chan string),
	}
}

func (s *outputStream) Write(buf []byte) (int, error) {
	s.stream <- string(buf)
	return len(buf), nil
}
