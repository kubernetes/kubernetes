// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package docker

import "testing"

func TestEnsureThinLsKernelVersion(t *testing.T) {
	tests := []struct {
		version       string
		expectedError string
	}{
		{"4.4.0-31-generic", ""},
		{"4.4.1", ""},
		{"4.6.4-301.fc24.x86_64", ""},
		{"3.10.0-327.22.2.el7.x86_64", `RHEL/Centos 7.x kernel version 3.10.0-366 or later is required to use thin_ls - you have "3.10.0-327.22.2.el7.x86_64"`},
		{"3.10.0-366.el7.x86_64", ""},
		{"3.10.0-366.el7_3.x86_64", ""},
		{"3.10.0.el7.abc", `unable to determine RHEL/Centos 7.x kernel release from "3.10.0.el7.abc"`},
		{"3.10.0-abc.el7.blarg", `unable to determine RHEL/Centos 7.x kernel release from "3.10.0-abc.el7.blarg"`},
		{"3.10.0-367.el7.x86_64", ""},
		{"3.10.0-366.x86_64", `kernel version 4.4.0 or later is required to use thin_ls - you have "3.10.0-366.x86_64"`},
		{"3.10.1-1.el7.x86_64", ""},
		{"2.0.36", `kernel version 4.4.0 or later is required to use thin_ls - you have "2.0.36"`},
		{"2.1", `error parsing kernel version: "2.1" is not a semver`},
	}

	for _, test := range tests {
		err := ensureThinLsKernelVersion(test.version)
		if err != nil {
			if len(test.expectedError) == 0 {
				t.Errorf("%s: expected no error, got %v", test.version, err)
			} else if err.Error() != test.expectedError {
				t.Errorf("%s: expected error %v, got %v", test.version, test.expectedError, err)
			}
		} else if err == nil && len(test.expectedError) > 0 {
			t.Errorf("%s: expected error %v", test.version, test.expectedError)
		}
	}
}

func TestIsContainerName(t *testing.T) {
	tests := []struct {
		name     string
		expected bool
	}{
		{
			name:     "/system.slice/var-lib-docker-overlay-9f086b233ab7c786bf8b40b164680b658a8f00e94323868e288d6ce20bc92193-merged.mount",
			expected: false,
		},
		{
			name:     "/system.slice/docker-72e5a5ff5eef3c4222a6551b992b9360a99122f77d2229783f0ee0946dfd800e.scope",
			expected: true,
		},
	}
	for _, test := range tests {
		if actual := isContainerName(test.name); actual != test.expected {
			t.Errorf("%s: expected: %v, actual: %v", test.name, test.expected, actual)
		}
	}
}
