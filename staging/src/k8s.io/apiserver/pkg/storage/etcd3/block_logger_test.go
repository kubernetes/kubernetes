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

package etcd3

import (
	"bytes"
	"flag"
	"fmt"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/klog/v2"
	testingclock "k8s.io/utils/clock/testing"
)

func TestBlockLogger(t *testing.T) {
	groupResource := schema.GroupResource{Group: "test", Resource: "pods"}
	msg := "Test message"

	tests := []struct {
		name          string
		interval      time.Duration
		threshold     time.Duration
		waits         []time.Duration
		timePassed    time.Duration
		expectLog     bool
		expectedCount int
		expectedWait  time.Duration
	}{
		{
			name:          "log when interval passed and threshold exceeded",
			interval:      1 * time.Second,
			threshold:     100 * time.Millisecond,
			waits:         []time.Duration{200 * time.Millisecond},
			timePassed:    2 * time.Second,
			expectLog:     true,
			expectedCount: 2, // 1 from waits + 1 from the triggering recordWait
			expectedWait:  200 * time.Millisecond,
		},
		{
			name:       "do not log when interval not passed",
			interval:   5 * time.Second,
			threshold:  100 * time.Millisecond,
			waits:      []time.Duration{200 * time.Millisecond},
			timePassed: 1 * time.Second,
			expectLog:  false,
		},
		{
			name:       "do not log when threshold not exceeded",
			interval:   1 * time.Second,
			threshold:  500 * time.Millisecond,
			waits:      []time.Duration{200 * time.Millisecond},
			timePassed: 2 * time.Second,
			expectLog:  false,
		},
		{
			name:          "aggregate multiple events",
			interval:      1 * time.Second,
			threshold:     100 * time.Millisecond,
			waits:         []time.Duration{100 * time.Millisecond, 200 * time.Millisecond},
			timePassed:    2 * time.Second,
			expectLog:     true,
			expectedCount: 3, // 2 from waits + 1 from the triggering recordWait
			expectedWait:  300 * time.Millisecond,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			flags := &flag.FlagSet{}
			klog.InitFlags(flags)
			if err := flags.Set("v", "3"); err != nil {
				t.Fatal(err)
			}
			klog.LogToStderr(false)
			klog.SetOutput(&buf)

			currentTime := time.Unix(0, 0)
			fakeClock := testingclock.NewFakeClock(currentTime)
			logger := newBlockLogger(tc.interval, tc.threshold, msg, "Pod", groupResource, fakeClock)

			// Record waits
			for _, w := range tc.waits {
				logger.recordWait(w)
			}

			// Simulate time passing
			fakeClock.Step(tc.timePassed)

			// Trigger another recordWait to check if it logs
			logger.recordWait(0)

			klog.Flush()
			logOutput := buf.String()
			logged := logOutput != ""

			if logged != tc.expectLog {
				t.Errorf("expected logged=%v, got %v. Output: %s", tc.expectLog, logged, logOutput)
			}

			if tc.expectLog {
				if !strings.Contains(logOutput, msg) {
					t.Errorf("expected log to contain message %q, got %q", msg, logOutput)
				}

				// Check for count and wait duration in the log output.
				countStr := fmt.Sprintf("eventCount=%d", tc.expectedCount)
				if !strings.Contains(logOutput, countStr) {
					t.Errorf("expected log to contain %q, got %q", countStr, logOutput)
				}

				waitStr := fmt.Sprintf("timeWaiting=%q", tc.expectedWait.String())
				if !strings.Contains(logOutput, waitStr) {
					t.Errorf("expected log to contain wait duration %q, got %q", waitStr, logOutput)
				}
			}
		})
	}
}
