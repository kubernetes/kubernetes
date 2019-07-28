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

package topologymanager

import (
	"testing"
)

func TestPolicyBestEffortCanAdmitPodResult(t *testing.T) {
	tcases := []struct {
		name     string
		admit    bool
		expected bool
	}{
		{
			name:     "Affinity is set to false in topology hints",
			admit:    false,
			expected: true,
		},
		{
			name:     "Affinity is set to true in topology hints",
			admit:    true,
			expected: true,
		},
	}

	for _, tc := range tcases {
		policy := NewBestEffortPolicy()
		admit := tc.admit
		result := policy.CanAdmitPodResult(admit)

		if result.Admit != tc.expected {
			t.Errorf("Expected Admit field in result to be %t, got %t", tc.expected, result.Admit)
		}
	}
}
