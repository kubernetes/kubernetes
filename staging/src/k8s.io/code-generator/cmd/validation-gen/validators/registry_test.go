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

package validators

import (
	"testing"
)

func TestGetStability(t *testing.T) {
	tests := []struct {
		tagName     string
		expected    TagStabilityLevel
		expectError bool
	}{
		{
			tagName:  "k8s:validateTrueAlpha",
			expected: TagStabilityLevelAlpha,
		},
		{
			tagName:  "k8s:validateTrueBeta",
			expected: TagStabilityLevelBeta,
		},
		{
			tagName:  "k8s:required",
			expected: TagStabilityLevelStable,
		},
		{
			tagName:     "k8s:unknownTag",
			expected:    "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.tagName, func(t *testing.T) {
			got, err := GetStability(tt.tagName)
			if err != nil && !tt.expectError {
				t.Errorf("Unexpected error: %v", err)
			}
			if got != tt.expected {
				t.Errorf("GetStability(%q) = %v, want %v", tt.tagName, got, tt.expected)
			}
		})
	}
}
