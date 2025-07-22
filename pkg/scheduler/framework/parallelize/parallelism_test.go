/*
Copyright 2020 The Kubernetes Authors.

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

package parallelize

import (
	"fmt"
	"testing"
)

func TestChunkSize(t *testing.T) {
	tests := []struct {
		input      int
		wantOutput int
	}{
		{
			input:      32,
			wantOutput: 3,
		},
		{
			input:      16,
			wantOutput: 2,
		},
		{
			input:      1,
			wantOutput: 1,
		},
		{
			input:      0,
			wantOutput: 1,
		},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%d", test.input), func(t *testing.T) {
			if chunkSizeFor(test.input, DefaultParallelism) != test.wantOutput {
				t.Errorf("Expected: %d, got: %d", test.wantOutput, chunkSizeFor(test.input, DefaultParallelism))
			}
		})
	}
}
