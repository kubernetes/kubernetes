/*
Copyright 2014 Google Inc. All rights reserved.

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

package gce_pd

import (
	"testing"
)

func TestGetDeviceName(t *testing.T) {
	tests := []struct {
		deviceName    string
		canonicalName string
		expectedName  string
		expectError   bool
	}{
		{
			deviceName:    "/dev/google-sd0-part0",
			canonicalName: "/dev/google/sd0P1",
			expectedName:  "sd0",
		},
		{
			canonicalName: "0123456",
			expectError:   true,
		},
	}
	for _, test := range tests {
		name, err := getDeviceName(test.deviceName, test.canonicalName)
		if test.expectError {
			if err == nil {
				t.Error("unexpected non-error")
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if name != test.expectedName {
			t.Errorf("expected: %s, got %s", test.expectedName, name)
		}
	}
}
