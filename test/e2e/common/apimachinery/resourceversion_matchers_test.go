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

package apimachinery

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestHaveValidResourceVersionMatch(t *testing.T) {
	tests := []struct {
		name            string
		resourceVersion string
		expectedMatch   bool
		expectedFailure string
	}{
		{
			name:            "valid resource version",
			resourceVersion: "123456789012345678901234567890123456789", // 128-bit unsigned int
			expectedMatch:   true,
		},
		{
			name:            "zero resource version",
			resourceVersion: "0",
			expectedMatch:   false,
			expectedFailure: "Expected resource version to be a valid uint128, but got \"0\": the resource version is zero which is not valid",
		},
		{
			name:            "negative resource version",
			resourceVersion: "-1",
			expectedMatch:   false,
			expectedFailure: "Expected resource version to be a valid uint128, but got \"-1\": the resource version is a negative number",
		},
		{
			name:            "non-numeric resource version",
			resourceVersion: "abc",
			expectedMatch:   false,
			expectedFailure: "Expected resource version to be a valid uint128, but got \"abc\": the resource version is not a valid integer",
		},
		{
			name:            "empty resource version",
			resourceVersion: "",
			expectedMatch:   false,
			expectedFailure: "Expected resource version to be a valid uint128, but got \"\": the resource version is not a valid integer",
		},
		{
			name:            "resource version too large (129 bits)",
			resourceVersion: "340282366920938463463374607431768211456", // 2^128
			expectedMatch:   false,
			expectedFailure: "Expected resource version to be a valid uint128, but got \"340282366920938463463374607431768211456\": resource version requires 129 bits (more than 128)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: tt.resourceVersion,
				},
			}

			matcher := HaveValidResourceVersion()
			match, err := matcher.Match(obj)

			if match != tt.expectedMatch {
				t.Errorf("Expected match to be %v, but got %v", tt.expectedMatch, match)
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if !match && matcher.FailureMessage(obj) != tt.expectedFailure {
				t.Errorf("Expected failure message to be %q, but got %q", tt.expectedFailure, matcher.FailureMessage(obj))
			}
		})
	}
}
