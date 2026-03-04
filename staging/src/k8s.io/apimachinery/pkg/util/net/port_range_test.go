/*
Copyright 2015 The Kubernetes Authors.

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

package net

import (
	"testing"

	flag "github.com/spf13/pflag"
)

func TestPortRange(t *testing.T) {
	testCases := []struct {
		input    string
		success  bool
		expected string
		included int
		excluded int
	}{
		{"100-200", true, "100-200", 200, 201},
		{" 100-200 ", true, "100-200", 200, 201},
		{"0-0", true, "0-0", 0, 1},
		{"", true, "", -1, 0},
		{"100", true, "100-100", 100, 101},
		{"100 - 200", false, "", -1, -1},
		{"-100", false, "", -1, -1},
		{"100-", false, "", -1, -1},
		{"200-100", false, "", -1, -1},
		{"60000-70000", false, "", -1, -1},
		{"70000-80000", false, "", -1, -1},
		{"70000+80000", false, "", -1, -1},
		{"1+0", true, "1-1", 1, 2},
		{"0+0", true, "0-0", 0, 1},
		{"1+-1", false, "", -1, -1},
		{"1-+1", false, "", -1, -1},
		{"100+200", true, "100-300", 300, 301},
		{"1+65535", false, "", -1, -1},
		{"0+65535", true, "0-65535", 65535, 65536},
	}

	for i := range testCases {
		tc := &testCases[i]
		pr := &PortRange{}
		var f flag.Value = pr
		err := f.Set(tc.input)
		if err != nil && tc.success {
			t.Errorf("expected success, got %q", err)
			continue
		} else if err == nil && !tc.success {
			t.Errorf("expected failure %#v", testCases[i])
			continue
		} else if tc.success {
			if f.String() != tc.expected {
				t.Errorf("expected %q, got %q", tc.expected, f.String())
			}
			if tc.included >= 0 && !pr.Contains(tc.included) {
				t.Errorf("expected %q to include %d", f.String(), tc.included)
			}
			if tc.excluded >= 0 && pr.Contains(tc.excluded) {
				t.Errorf("expected %q to exclude %d", f.String(), tc.excluded)
			}
		}
	}
}
