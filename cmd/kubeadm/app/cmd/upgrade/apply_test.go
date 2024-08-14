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

package upgrade

import (
	"testing"
)

func TestSessionIsInteractive(t *testing.T) {
	var tcases = []struct {
		name     string
		data     *applyData
		expected bool
	}{
		{
			name: "Explicitly non-interactive",
			data: &applyData{
				nonInteractiveMode: true,
			},
			expected: false,
		},
		{
			name: "Implicitly non-interactive since --dryRun is used",
			data: &applyData{
				dryRun: true,
			},
			expected: false,
		},
		{
			name: "Implicitly non-interactive since --force is used",
			data: &applyData{
				force: true,
			},
			expected: false,
		},
		{
			name:     "Interactive session",
			data:     &applyData{},
			expected: true,
		},
	}
	for _, tt := range tcases {
		t.Run(tt.name, func(t *testing.T) {
			if tt.data.SessionIsInteractive() != tt.expected {
				t.Error("unexpected result")
			}
		})
	}
}
