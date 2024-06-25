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
	"context"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
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

func TestParallelizeUntil(t *testing.T) {
	tests := []struct {
		pieces    int
		chunkSize int
	}{
		{
			pieces:    1000,
			chunkSize: 1,
		},
		{
			pieces:    1000,
			chunkSize: 10,
		},
		{
			pieces:    1000,
			chunkSize: 100,
		},
		{
			pieces:    999,
			chunkSize: 13,
		},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf("pieces:%v,chunkSize:%v", tc.pieces, tc.chunkSize), func(t *testing.T) {
			seen := make([]int32, tc.pieces)
			parallelizer := NewParallelizer(DefaultParallelism)
			parallelizer.Until(context.Background(), tc.pieces, func(p int) {
				atomic.AddInt32(&seen[p], 1)
			}, "test-parallelize-until")

			wantSeen := make([]int32, tc.pieces)
			for i := 0; i < tc.pieces; i++ {
				wantSeen[i] = 1
			}

			if diff := cmp.Diff(wantSeen, seen); diff != "" {
				t.Errorf("bad number of visits (-want,+got):\n%s", diff)
			}
		})
	}
}
