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

package gce

import "testing"

func TestIsPortsSubset(t *testing.T) {
	tc := map[string]struct {
		required  []string
		coverage  []string
		expectErr bool
	}{
		"Single port coverage": {
			required: []string{"tcp/50"},
			coverage: []string{"tcp/50", "tcp/60", "tcp/70"},
		},
		"Port range coverage": {
			required: []string{"tcp/50"},
			coverage: []string{"tcp/20-30", "tcp/45-60"},
		},
		"Multiple Port range coverage": {
			required: []string{"tcp/50", "tcp/29", "tcp/46"},
			coverage: []string{"tcp/20-30", "tcp/45-60"},
		},
		"Not covered": {
			required:  []string{"tcp/50"},
			coverage:  []string{"udp/50", "tcp/49", "tcp/51-60"},
			expectErr: true,
		},
	}

	for name, c := range tc {
		t.Run(name, func(t *testing.T) {
			gotErr := isPortsSubset(c.required, c.coverage)
			if c.expectErr != (gotErr != nil) {
				t.Errorf("isPortsSubset(%v, %v) = %v, wanted err? %v", c.required, c.coverage, gotErr, c.expectErr)
			}
		})
	}
}
