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

package queue

import (
	"fmt"
	"testing"
	"time"

	ktypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
)

func TestBackoffPod(t *testing.T) {
	timestamp := time.Now()
	bpm := NewPodBackoffMap(1*time.Second, 10*time.Second, clock.NewFakeClock(timestamp))
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
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 8 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 10 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 10 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "bar"},
			expectedDuration: 1 * time.Second,
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("step %d", i), func(t *testing.T) {
			bpm.BackoffPod(test.podID)
			backoff, ok := bpm.GetBackoffTime(test.podID)
			if !ok {
				t.Errorf("%v should be backed off", test.podID)
			}
			duration := backoff.Sub(timestamp)
			if duration != test.expectedDuration {
				t.Errorf("expected: %s, got %s for pod %s", test.expectedDuration.String(), duration.String(), test.podID)
			}
		})
	}
}

func TestClearPodBackoff(t *testing.T) {
	bpm := NewPodBackoffMap(1*time.Second, 60*time.Second, clock.NewFakeClock(time.Now()))
	// Clear backoff on an not existed pod
	bpm.clearPodBackoff(ktypes.NamespacedName{Namespace: "ns", Name: "not-existed"})
	// Backoff twice for pod foo
	podID := ktypes.NamespacedName{Namespace: "ns", Name: "foo"}
	bpm.BackoffPod(podID)
	bpm.BackoffPod(podID)
	if duration := bpm.calculateBackoffDuration(podID); duration != 2*time.Second {
		t.Errorf("Expected backoff of 1s for pod %s, got %s", podID, duration.String())
	}
	// Clear backoff for pod foo
	bpm.clearPodBackoff(podID)
	// Backoff once for pod foo
	bpm.BackoffPod(podID)
	if duration := bpm.calculateBackoffDuration(podID); duration != 1*time.Second {
		t.Errorf("Expected backoff of 1s for pod %s, got %s", podID, duration.String())
	}
}
