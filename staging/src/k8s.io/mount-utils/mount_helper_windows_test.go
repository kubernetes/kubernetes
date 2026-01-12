//go:build windows
// +build windows

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

package mount

import (
	"testing"
)

func TestNormalizeWindowsPath(t *testing.T) {
	path := `/var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4/volumes/kubernetes.io~azure-disk`
	normalizedPath := NormalizeWindowsPath(path)
	if normalizedPath != `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}

	path = `/var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk`
	normalizedPath = NormalizeWindowsPath(path)
	if normalizedPath != `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}

	path = `/`
	normalizedPath = NormalizeWindowsPath(path)
	if normalizedPath != `c:\` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}
}

func TestValidateDiskNumber(t *testing.T) {
	tests := []struct {
		diskNum     string
		expectError bool
	}{
		{
			diskNum:     "0",
			expectError: false,
		},
		{
			diskNum:     "invalid",
			expectError: true,
		},
		{
			diskNum:     "99",
			expectError: false,
		},
		{
			diskNum:     "100",
			expectError: false,
		},
		{
			diskNum:     "200",
			expectError: false,
		},
	}

	for _, test := range tests {
		err := ValidateDiskNumber(test.diskNum)
		if (err != nil) != test.expectError {
			t.Errorf("TestValidateDiskNumber test failed, disk number: %s, error: %v", test.diskNum, err)
		}
	}
}
