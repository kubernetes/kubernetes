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

func TestBackoff(t *testing.T) {
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
		duration := backoff.GetEntry(test.podID).getBackoff(backoff.maxDuration)
		if duration != test.expectedDuration {
			t.Errorf("expected: %s, got %s for %s", test.expectedDuration.String(), duration.String(), test.podID)
		}
		clock.t = clock.t.Add(test.advanceClock)
		backoff.Gc()
	}
	fooID := ktypes.NamespacedName{Namespace: "default", Name: "foo"}
	backoff.perPodBackoff[fooID].backoff = 60 * time.Second
	duration := backoff.GetEntry(fooID).getBackoff(backoff.maxDuration)
	if duration != 60*time.Second {
		t.Errorf("expected: 60, got %s", duration.String())
	}
	// Verify that we split on namespaces correctly, same name, different namespace
	fooID.Namespace = "other"
	duration = backoff.GetEntry(fooID).getBackoff(backoff.maxDuration)
	if duration != 1*time.Second {
		t.Errorf("expected: 1, got %s", duration.String())
	}
}
