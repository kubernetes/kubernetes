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
	"bytes"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
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
			expectLog: `<klog header>: step: Log a b 42
<klog header>: step: Logf a b 42
`,
			expectError: `step: Error a b 42
step: Errorf a b 42`,
		},
		"fatal": {
			cb: func(tCtx TContext) {
				tCtx = WithStep(tCtx, "step")
				tCtx.Fatal("Error", "a", "b", 42)
				// not reached
				tCtx.Log("Log")
			},
			expectError: `step: Error a b 42`,
		},
		"fatalf": {
			cb: func(tCtx TContext) {
				tCtx = WithStep(tCtx, "step")
				tCtx.Fatalf("Error %s %s %d", "a", "b", 42)
				// not reached
				tCtx.Log("Log")
			},
			expectError: `step: Error a b 42`,
		},
		"progress": {
			cb: func(tCtx TContext) {
				tCtx = WithStep(tCtx, "step")
				var buffer bytes.Buffer
				oldOut := defaultProgressReporter.setOutput(&buffer)
				defer defaultProgressReporter.setOutput(oldOut)
				remove := tCtx.Value("GINKGO_SPEC_CONTEXT").(ginkgoReporter).AttachProgressReporter(func() string { return "hello world" })
				defer remove()
				defaultSignalChannel <- os.Interrupt
				// No good way to sync here, so let's just wait.
				time.Sleep(5 * time.Second)
				defaultProgressReporter.setOutput(oldOut)
				tCtx.Log(buffer.String())

				noSuchValue := tCtx.Value("some other key")
				assert.Equal(tCtx, nil, noSuchValue, "value for unknown context value key")
			},
			expectLog: `<klog header>: step: You requested a progress report.

step: hello world
`,
			expectDuration: 5 * time.Second,
			expectNoFail:   true,
		},
	} {
		tc := tc
		t.Run(name, func(t *testing.T) {
			tc.run(t)
		})
	}
}
