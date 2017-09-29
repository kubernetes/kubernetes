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

package preflight

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

func TestGetKubeletVersion(t *testing.T) {
	type T struct {
		output   string
		expected string
		valid    bool
	}

	cases := []T{
		{"v1.7.0", "1.7.0", true},
		{"v1.8.0-alpha.2.1231+afabd012389d53a", "1.8.0-alpha.2.1231+afabd012389d53a", true},
		{"something-invalid", "", false},
	}

	dir, err := ioutil.TempDir("", "test-kubelet-version")
	if err != nil {
		t.Errorf("Failed to create directory for testing GetKubeletVersion: %v", err)
	}
	defer os.RemoveAll(dir)

	// We don't want to call real kubelet or something else in $PATH
	oldPATH := os.Getenv("PATH")
	defer os.Setenv("PATH", oldPATH)

	os.Setenv("PATH", dir)

	// First test case, kubelet not present, should be getting error
	ver, err := GetKubeletVersion()
	if err == nil {
		t.Errorf("failed GetKubeletVersion: expected failure when kubelet not in PATH. Result: %v", ver)
	}

	kubeletFn := filepath.Join(dir, "kubelet")
	for _, tc := range cases {

		content := []byte(fmt.Sprintf("#!/bin/sh\necho 'Kubernetes %s'", tc.output))
		if err := ioutil.WriteFile(kubeletFn, content, 0755); err != nil {
			t.Errorf("Error creating test stub file %s: %v", kubeletFn, err)
		}

		ver, err := GetKubeletVersion()
		switch {
		case err != nil && tc.valid:
			t.Errorf("GetKubeletVersion: unexpected error for %q. Error: %v", tc.output, err)
		case err == nil && !tc.valid:
			t.Errorf("GetKubeletVersion: error expected for key %q, but result is %q", tc.output, ver)
		case ver != nil && ver.String() != tc.expected:
			t.Errorf("GetKubeletVersion: unexpected version result for key %q. Expected: %q Actual: %q", tc.output, tc.expected, ver)
		}

	}

}
