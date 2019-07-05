/*
Copyright 2018 The Kubernetes Authors.

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
)

func TestGetCgroupDriverDocker(t *testing.T) {
	testCases := []struct {
		name          string
		driver        string
		expectedError bool
	}{
		{
			name:          "valid: value is 'cgroupfs'",
			driver:        `cgroupfs`,
			expectedError: false,
		},
		{
			name:          "valid: value is 'systemd'",
			driver:        `systemd`,
			expectedError: false,
		},
		{
			name:          "invalid: empty 'Cgroup Driver' value",
			driver:        ``,
			expectedError: true,
		},
		{
			name:          "invalid: unknown 'Cgroup Driver' value",
			driver:        `invalid-value`,
			expectedError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := tc.driver != CgroupDriverCgroupfs && tc.driver != CgroupDriverSystemd
			if result != tc.expectedError {
				t.Fatalf("expected error: %v, saw: %v", tc.expectedError, result)
			}
		})
	}
}
