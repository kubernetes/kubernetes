/*
Copyright 2025 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestOptions_ControllerListFormatter(t *testing.T) {
	testcases := []struct {
		name        string
		descriptors map[string]*ControllerDescriptor
		controllers []string
		expected    string
	}{
		{
			name:        "unknown controller name",
			descriptors: nil,
			controllers: []string{"controller"},
			expected:    "\ncontroller\n",
		},
		{
			name: "without feature gates",
			descriptors: map[string]*ControllerDescriptor{
				"controller_A": {},
				"controller_B": {},
			},
			controllers: []string{"controller_A", "controller_B"},
			expected:    "\ncontroller_A\ncontroller_B\n",
		},
		{
			name: "with feature gates",
			descriptors: map[string]*ControllerDescriptor{
				"controller_A": {},
				"controller_B": {
					requiredFeatureGates: []string{
						"F1",
						"F2",
					},
				},
				"controller_C": {},
			},
			controllers: []string{"controller_A", "controller_B"},
			expected:    "\ncontroller_A\ncontroller_B (requires feature gates: F1, F2)\n",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			got := newControllerListFormatter(tc.descriptors)(tc.controllers)
			if got != tc.expected {
				t.Error("unexpected output:\n", cmp.Diff(tc.expected, got))
			}
		})
	}
}

func TestCommand_ControllerListFormatter(t *testing.T) {
	// This test makes sure the formatter is actually set properly.
	// This can break, though, when no feature gate is actually required.
	cmd := NewControllerManagerCommand()
	usage := cmd.Flags().Lookup("controllers").Usage
	if !strings.Contains(usage, "(requires feature gates:") {
		t.Error("unexpected usage:\n", usage)
	}
}
