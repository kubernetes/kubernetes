/*
Copyright 2024 The Kubernetes Authors.

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

package swap

import "testing"

func TestIsSwapEnabled(t *testing.T) {
	testCases := []struct {
		name             string
		procSwapsContent string
		expectedEnabled  bool
	}{
		{
			name:             "empty",
			procSwapsContent: "",
			expectedEnabled:  false,
		},
		{
			name: "with swap enabled, one partition",
			procSwapsContent: `
Filename				Type		Size		Used		Priority
/dev/dm-1               partition	33554428	0		-2
`,
			expectedEnabled: true,
		},
		{
			name: "with swap enabled, 2 partitions",
			procSwapsContent: `
Filename				Type		Size		Used		Priority
/dev/dm-1               partition	33554428	0		-2
/dev/zram0              partition	8388604		0		100
`,
			expectedEnabled: true,
		},
		{
			name: "empty lines",
			procSwapsContent: `

`,
			expectedEnabled: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			isEnabled := isSwapOnAccordingToProcSwaps([]byte(tc.procSwapsContent))
			if isEnabled != tc.expectedEnabled {
				t.Errorf("expected %v, got %v", tc.expectedEnabled, isEnabled)
			}
		})
	}
}
