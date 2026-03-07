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

package prober

import (
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/test/utils/ktesting"

	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

// TestFailureThresholdEvents verifies that an event is emitted even when a probe failure
// is suppressed by FailureThreshold.
func TestFailureThresholdEvents(t *testing.T) {
	// Register types in the scheme so GenerateContainerRef works
	if legacyscheme.Scheme == nil {
		t.Fatal("Scheme is nil")
	}

	logger, ctx := ktesting.NewTestContext(t)
	m := newTestManager()

	// Use a REAL FakeRecorder with a buffer
	fakeRecorder := record.NewFakeRecorder(100)
	m.prober.recorder = fakeRecorder

	// Create a Liveness probe with FailureThreshold = 3
	w := newTestWorker(m, liveness, v1.Probe{
		SuccessThreshold: 1,
		FailureThreshold: 3,
	})

	// Ensure the pod is "running" so probing happens
	w.pod.Namespace = "default"
	w.pod.ResourceVersion = "1234"
	m.statusManager.SetPodStatus(logger, w.pod, getTestRunningStatus())

	// 1. First Probe: SUCCESS
	m.prober.exec = fakeExecProber{probe.Success, nil}
	w.doProbe(ctx)

	// Clear any events from the success
	clearEvents(fakeRecorder)

	// 2. Second Probe: FAILURE (1/3)
	m.prober.exec = fakeExecProber{probe.Failure, nil}
	w.doProbe(ctx)

	// Give the async recorder a tiny bit of time
	time.Sleep(100 * time.Millisecond)

	// Verify we got the SPECIFIC suppression event
	found := false
	eventCount := 0
	for {
		select {
		case event := <-fakeRecorder.Events:
			eventCount++
			t.Logf("Event #%d Received: %s", eventCount, event)
			if strings.Contains(event, "suppressed") || strings.Contains(event, "ignored") {
				found = true
			}
		default:
			goto Done
		}
	}
Done:
	t.Logf("Total events received: %d", eventCount)
	if !found {
		t.Error("Expected an event indicating the failure was suppressed/ignored, but got none.")
	}
}

func clearEvents(r *record.FakeRecorder) {
	for {
		select {
		case <-r.Events:
		default:
			return
		}
	}
}
