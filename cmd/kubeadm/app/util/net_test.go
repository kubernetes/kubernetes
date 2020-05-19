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

package util

import (
	"errors"
	"os"
	"strings"
	"testing"
)

func TestGetHostname(t *testing.T) {
	hostname, err := os.Hostname()

	testCases := []struct {
		desc        string
		hostname    string
		result      string
		expectedErr error
	}{
		{
			desc:        "overridden hostname",
			hostname:    "overridden",
			result:      "overridden",
			expectedErr: nil,
		},
		{
			desc:        "overridden hostname uppercase",
			hostname:    "OVERRIDDEN",
			result:      "overridden",
			expectedErr: nil,
		},
		{
			desc:        "hostname contains only spaces",
			hostname:    " ",
			result:      "",
			expectedErr: errors.New("empty hostname is invalid"),
		},
		{
			desc:        "empty parameter",
			hostname:    "",
			result:      strings.ToLower(hostname),
			expectedErr: err,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			result, err := GetHostname(tc.hostname)

			if err != nil && tc.expectedErr == nil {
				t.Errorf("unexpected error: %v", err)
			}

			if err == nil && tc.expectedErr != nil {
				t.Errorf("expected error %v, got nil", tc.expectedErr)
			}

			if tc.result != result {
				t.Errorf("unexpected result: %s, expected: %s", result, tc.result)
			}
		})
	}
}
