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
	"context"
	"io"
	"os"
	"testing"
	"time"

	"github.com/onsi/gomega"
	"go.uber.org/goleak"
)

func TestStepContext(t *testing.T) {
	for name, tc := range map[string]testcase{
		"output": {
			cb: func(tCtx TContext) {
				tCtx = WithStep(tCtx, "step")
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
		"fatal": {
			cb: func(tCtx TContext) {
				tCtx = WithStep(tCtx, "step")
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
				tCtx = WithStep(tCtx, "step")
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
	tCtx = WithStep(tCtx, "step")
	removeReporter := tCtx.Value("GINKGO_SPEC_CONTEXT").(ginkgoReporter).AttachProgressReporter(func() string { return "hello world" })
	defer removeReporter()
	tCtx.Expect(tCtx.Value("some other key")).To(gomega.BeNil(), "value for unknown context value key")

	// Trigger report and wait for it.
	defaultProgressReporter.progressChannel <- os.Interrupt
	report := <-out.stream
	tCtx.Expect(report).To(gomega.Equal(`You requested a progress report.

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
