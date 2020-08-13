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

package metrics

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDefaultStabilityLevel(t *testing.T) {
	var tests = []struct {
		name        string
		inputValue  StabilityLevel
		expectValue StabilityLevel
		expectPanic bool
	}{
		{
			name:        "empty should take ALPHA by default",
			inputValue:  "",
			expectValue: ALPHA,
			expectPanic: false,
		},
		{
			name:        "ALPHA remain unchanged",
			inputValue:  ALPHA,
			expectValue: ALPHA,
			expectPanic: false,
		},
		{
			name:        "STABLE remain unchanged",
			inputValue:  STABLE,
			expectValue: STABLE,
			expectPanic: false,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			var stability = tc.inputValue

			stability.setDefaults()
			assert.Equalf(t, tc.expectValue, stability, "Got %s, expected: %v ", stability, tc.expectValue)
		})
	}
}
