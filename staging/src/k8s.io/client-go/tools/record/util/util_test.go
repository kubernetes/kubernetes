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

package util

import (
	"strings"
	"sync"
	"testing"

	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
)

func TestGenerateEventName(t *testing.T) {
	timestamp := int64(105999103295324396)
	testCases := []struct {
		name           string
		refName        string
		expectedPrefix string
	}{
		{
			name:           "valid name",
			refName:        "test-pod",
			expectedPrefix: "test-pod.178959f726d80ec.",
		},
		{
			name:    "invalid name - too long",
			refName: strings.Repeat("x", 300),
		},
		{
			name:    "invalid name - upper case",
			refName: "test.POD",
		},
		{
			name:    "invalid name - special chars",
			refName: "test.pod/invalid!chars?",
		},
		{
			name:    "invalid name - special chars and non alphanumeric starting character",
			refName: "--test.pod/invalid!chars?",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := GenerateEventName(tc.refName, timestamp)

			if errs := apimachineryvalidation.NameIsDNSSubdomain(actual, false); len(errs) > 0 {
				t.Errorf("generateEventName(%s) = %s; not a valid name: %v", tc.refName, actual, errs)

			}

			if tc.expectedPrefix != "" && !strings.HasPrefix(actual, tc.expectedPrefix) {
				t.Errorf("generateEventName(%s) returned %s expected prefix %s", tc.refName, actual, tc.expectedPrefix)
			}

		})

	}
}

// TestGenerateEventNameUniqueOnCollidingTimestamp guards against a regression
// where two events for the same object, generated within the same timer tick
// (e.g. on platforms with coarse timer resolution such as Windows), computed
// to the same Event name and silently collided on the apiserver.
// See https://github.com/kubernetes/kubernetes/issues/134993.
func TestGenerateEventNameUniqueOnCollidingTimestamp(t *testing.T) {
	timestamp := int64(105999103295324396)

	first := GenerateEventName("test-pod", timestamp)
	second := GenerateEventName("test-pod", timestamp)

	if first == second {
		t.Errorf("GenerateEventName produced the same name %q for two events sharing a timestamp", first)
	}
}

// TestGenerateEventNameConcurrentUnique verifies GenerateEventName is safe to
// call concurrently, which is the actual calling pattern from EventRecorder
// implementations: controllers record events from arbitrary goroutines.
func TestGenerateEventNameConcurrentUnique(t *testing.T) {
	const n = 1000
	names := make([]string, n)
	var wg sync.WaitGroup
	for i := range names {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			names[i] = GenerateEventName("test-pod", 105999103295324396)
		}(i)
	}
	wg.Wait()

	seen := make(map[string]bool, n)
	for _, name := range names {
		if seen[name] {
			t.Fatalf("duplicate generated name %q under concurrent calls", name)
		}
		seen[name] = true
	}
}
