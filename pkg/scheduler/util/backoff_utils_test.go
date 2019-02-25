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

package util

import (
	"testing"
	"time"

	ktypes "k8s.io/apimachinery/pkg/types"
)

type fakeClock struct {
	t time.Time
}

func (f *fakeClock) Now() time.Time {
	return f.t
}

func TestBackoffPod(t *testing.T) {
	clock := fakeClock{}
	backoff := CreatePodBackoffWithClock(1*time.Second, 60*time.Second, &clock)
	tests := []struct {
		podID            ktypes.NamespacedName
		expectedDuration time.Duration
		advanceClock     time.Duration
	}{
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 1 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 2 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 4 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "bar"},
			expectedDuration: 1 * time.Second,
			advanceClock:     120 * time.Second,
		},
		// 'foo' should have been gc'd here.
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 1 * time.Second,
		},
	}

	for _, test := range tests {
		duration := backoff.BackoffPod(test.podID)
		if duration != test.expectedDuration {
			t.Errorf("expected: %s, got %s for pod %s", test.expectedDuration.String(), duration.String(), test.podID)
		}
		if boTime, _ := backoff.GetBackoffTime(test.podID); boTime != clock.Now().Add(test.expectedDuration) {
			t.Errorf("expected GetBackoffTime %s, got %s for pod %s", test.expectedDuration.String(), boTime.String(), test.podID)
		}
		clock.t = clock.t.Add(test.advanceClock)
		backoff.Gc()
	}
	fooID := ktypes.NamespacedName{Namespace: "default", Name: "foo"}
	be := backoff.getEntry(fooID)
	be.backoff = 60 * time.Second
	duration := backoff.BackoffPod(fooID)
	if duration != 60*time.Second {
		t.Errorf("expected: 60, got %s", duration.String())
	}
	// Verify that we split on namespaces correctly, same name, different namespace
	fooID.Namespace = "other"
	duration = backoff.BackoffPod(fooID)
	if duration != 1*time.Second {
		t.Errorf("expected: 1, got %s", duration.String())
	}
}

func TestClearPodBackoff(t *testing.T) {
	clock := fakeClock{}
	backoff := CreatePodBackoffWithClock(1*time.Second, 60*time.Second, &clock)

	if backoff.ClearPodBackoff(ktypes.NamespacedName{Namespace: "ns", Name: "nonexist"}) {
		t.Error("Expected ClearPodBackoff failure for unknown pod, got success.")
	}

	podID := ktypes.NamespacedName{Namespace: "ns", Name: "foo"}
	if dur := backoff.BackoffPod(podID); dur != 1*time.Second {
		t.Errorf("Expected backoff of 1s for pod %s, got %s", podID, dur.String())
	}

	if !backoff.ClearPodBackoff(podID) {
		t.Errorf("Failed to clear backoff for pod %v", podID)
	}

	expectBoTime := clock.Now()
	if boTime, _ := backoff.GetBackoffTime(podID); boTime != expectBoTime {
		t.Errorf("Expected backoff time for pod %s of %s, got %s", podID, expectBoTime, boTime)
	}
}

func TestTryBackoffAndWait(t *testing.T) {
	clock := fakeClock{}
	backoff := CreatePodBackoffWithClock(1*time.Second, 60*time.Second, &clock)

	stopCh := make(chan struct{})
	podID := ktypes.NamespacedName{Namespace: "ns", Name: "pod"}
	if !backoff.TryBackoffAndWait(podID, stopCh) {
		t.Error("Expected TryBackoffAndWait success for new pod, got failure.")
	}

	be := backoff.getEntry(podID)
	if !be.tryLock() {
		t.Error("Failed to acquire lock for backoffentry")
	}

	if backoff.TryBackoffAndWait(podID, stopCh) {
		t.Error("Expected TryBackoffAndWait failure with lock acquired, got success.")
	}

	close(stopCh)
	if backoff.TryBackoffAndWait(podID, stopCh) {
		t.Error("Expected TryBackoffAndWait failure with closed stopCh, got success.")
	}
}
