/*
Copyright 2021 The Kubernetes Authors.

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

package options

import "testing"

func TestValidateTracingOptions(t *testing.T) {
	testcases := []struct {
		name        string
		expectError bool
		contents    *TracingOptions
	}{
		{
			name:        "nil-valid",
			expectError: false,
		},
		{
			name:        "empty-valid",
			expectError: false,
			contents:    &TracingOptions{},
		},
		{
			name:        "path-valid",
			expectError: false,
			contents:    &TracingOptions{ConfigFile: "/"},
		},
		{
			name:        "path-invalid",
			expectError: true,
			contents:    &TracingOptions{ConfigFile: "/path/doesnt/exist"},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			errs := tc.contents.Validate()
			if tc.expectError == false && len(errs) != 0 {
				t.Errorf("Calling Validate expected no error, got %v", errs)
			} else if tc.expectError == true && len(errs) == 0 {
				t.Errorf("Calling Validate expected error, got no error")
			}
		})
	}
}
