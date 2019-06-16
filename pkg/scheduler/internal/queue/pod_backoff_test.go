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
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ktypes "k8s.io/apimachinery/pkg/types"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

var pod1 = &v1.Pod{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "test-pod-1",
		Namespace: "ns1",
		UID:       ktypes.UID("tp-1"),
	},
	Status: v1.PodStatus{
		NominatedNodeName: "node1",
	},
}

var timestamp = time.Now()
var podInfo1 = &framework.PodInfo{
	Pod:       pod1,
	Timestamp: timestamp,
}

// TestBackoffPod tests pod backoff duration
func TestBackoffPod(t *testing.T) {
	bpm := NewPodBackoffMap(1*time.Second, 10*time.Second)

	tests := []struct {
		podID            ktypes.NamespacedName
		podInfo          *framework.PodInfo
		expectedDuration time.Duration
		advanceClock     time.Duration
	}{
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			podInfo:          podInfo1,
			expectedDuration: 1 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			podInfo:          podInfo1,
			expectedDuration: 2 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			podInfo:          podInfo1,
			expectedDuration: 4 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			podInfo:          podInfo1,
			expectedDuration: 8 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			podInfo:          podInfo1,
			expectedDuration: 10 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "foo"},
			podInfo:          podInfo1,
			expectedDuration: 10 * time.Second,
		},
		{
			podID:            ktypes.NamespacedName{Namespace: "default", Name: "bar"},
			podInfo:          podInfo1,
			expectedDuration: 1 * time.Second,
		},
	}

	for _, test := range tests {
		// Backoff the pod
		bpm.BackoffPod(test.podID, podInfo1)
		// Get backoff duration for the pod
		duration := bpm.calculateBackoffDuration(test.podID)

		if duration != test.expectedDuration {
			t.Errorf("expected: %s, got %s for pod %s", test.expectedDuration.String(), duration.String(), test.podID)
		}
	}
}

// TestClearPodBackoff tests clearance of pod backoff data in PodBackoffMap
func TestClearPodBackoff(t *testing.T) {
	bpm := NewPodBackoffMap(1*time.Second, 60*time.Second)
	// Clear backoff on an not existed pod
	bpm.clearPodBackoff(ktypes.NamespacedName{Namespace: "ns", Name: "not-existed"})
	// Backoff twice for pod foo
	podID := ktypes.NamespacedName{Namespace: "ns", Name: "foo"}
	bpm.BackoffPod(podID, podInfo1)
	bpm.BackoffPod(podID, podInfo1)
	if duration := bpm.calculateBackoffDuration(podID); duration != 2*time.Second {
		t.Errorf("Expected backoff of 1s for pod %s, got %s", podID, duration.String())
	}
	// Clear backoff for pod foo
	bpm.clearPodBackoff(podID)
	// Backoff once for pod foo
	bpm.BackoffPod(podID, podInfo1)
	if duration := bpm.calculateBackoffDuration(podID); duration != 1*time.Second {
		t.Errorf("Expected backoff of 1s for pod %s, got %s", podID, duration.String())
	}
}
