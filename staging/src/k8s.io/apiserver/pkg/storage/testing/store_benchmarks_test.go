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

package testing

import (
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/apis/example"
	testingclock "k8s.io/utils/clock/testing"
)

func TestWatchLatencyTimestampRoundTrip(t *testing.T) {
	now := time.Unix(0, time.Now().UnixNano()) // Round to nano precision
	serialized := serializeTimestamp(now)
	parsed, err := parseTimestamp(serialized)
	if err != nil {
		t.Fatalf("Failed to parse timestamp: %v", err)
	}
	if !parsed.Equal(now) {
		t.Errorf("Round-trip failed: got %v, expected %v", parsed, now)
	}
}

func TestWatchLatencyTracker_RecordWriteAndHandleEvent(t *testing.T) {
	initialTime := time.Date(2026, 6, 7, 9, 0, 0, 0, time.UTC)
	fakeClock := testingclock.NewFakeClock(initialTime)
	tracker := NewWatchLatencyTracker(fakeClock)

	// Record 100 events:
	// - 98 events (1 to 98) with 10ms delay
	// - 1 event (99th) with 50ms delay (this is the 99%ile)
	// - 1 event (100th, which is the max) with 100ms delay
	for i := 1; i <= 100; i++ {
		pod := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("pod-%d", i),
			},
		}
		tracker.RecordWrite(pod)

		var delay time.Duration
		switch i {
		case 99:
			delay = 50 * time.Millisecond
		case 100:
			delay = 100 * time.Millisecond
		default:
			delay = 10 * time.Millisecond
		}
		fakeClock.Step(delay)
		tracker.HandleEvent(pod)

		// Reset back to base fake clock time for the next pod so it's not cumulative
		fakeClock.SetTime(initialTime)
	}

	p99 := tracker.GetP99Latency()
	if p99 != 50*time.Millisecond {
		t.Errorf("Expected P99 watch latency to be 50ms, got %v (max was 100ms)", p99)
	}
}
