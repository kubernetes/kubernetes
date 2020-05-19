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

package app

import (
	"testing"
)

func TestGetServiceIPAndRanges(t *testing.T) {
	tests := []struct {
		body                    string
		apiServerServiceIP      string
		primaryServiceIPRange   string
		secondaryServiceIPRange string
		expectedError           bool
	}{
		{"", "10.0.0.1", "10.0.0.0/24", "<nil>", false},
		{"192.0.2.1/24", "192.0.2.1", "192.0.2.0/24", "<nil>", false},
		{"192.0.2.1/24,192.168.128.0/17", "192.0.2.1", "192.0.2.0/24", "192.168.128.0/17", false},
		{"192.0.2.1/30,192.168.128.0/17", "<nil>", "<nil>", "<nil>", true},
	}

	for _, test := range tests {
		apiServerServiceIP, primaryServiceIPRange, secondaryServiceIPRange, err := getServiceIPAndRanges(test.body)

		if apiServerServiceIP.String() != test.apiServerServiceIP {
			t.Errorf("expected apiServerServiceIP: %s, got: %s", test.apiServerServiceIP, apiServerServiceIP.String())
		}

		if primaryServiceIPRange.String() != test.primaryServiceIPRange {
			t.Errorf("expected primaryServiceIPRange: %s, got: %s", test.primaryServiceIPRange, primaryServiceIPRange.String())
		}

		if secondaryServiceIPRange.String() != test.secondaryServiceIPRange {
			t.Errorf("expected secondaryServiceIPRange: %s, got: %s", test.secondaryServiceIPRange, secondaryServiceIPRange.String())
		}

		if (err == nil) == test.expectedError {
			t.Errorf("expected err to be: %t, but it was %t", test.expectedError, !test.expectedError)
		}
	}
}
