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

package workqueue

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
)

type testCase struct {
	pieces    int
	workers   int
	chunkSize int
}

func (c testCase) String() string {
	return fmt.Sprintf("pieces:%d,workers:%d,chunkSize:%d", c.pieces, c.workers, c.chunkSize)
}

var cases = []testCase{
	{
		pieces:    1000,
		workers:   10,
		chunkSize: 1,
	},
	{
		pieces:    1000,
		workers:   10,
		chunkSize: 10,
	},
	{
		pieces:    1000,
		workers:   10,
		chunkSize: 100,
	},
	{
		pieces:    999,
		workers:   10,
		chunkSize: 13,
	},
}

func TestParallelizeUntil(t *testing.T) {
	for _, tc := range cases {
		t.Run(tc.String(), func(t *testing.T) {
			seen := make([]int32, tc.pieces)
			ctx := context.Background()
			ParallelizeUntil(ctx, tc.workers, tc.pieces, func(p int) {
				atomic.AddInt32(&seen[p], 1)
			}, WithChunkSize(tc.chunkSize))

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

func BenchmarkParallelizeUntil(b *testing.B) {
	for _, tc := range cases {
		b.Run(tc.String(), func(b *testing.B) {
			ctx := context.Background()
			isPrime := make([]bool, tc.pieces)
			b.ResetTimer()
			for c := 0; c < b.N; c++ {
				ParallelizeUntil(ctx, tc.workers, tc.pieces, func(p int) {
					isPrime[p] = calPrime(p)
				}, WithChunkSize(tc.chunkSize))
			}
			b.StopTimer()
			want := []bool{false, false, true, true, false, true, false, true, false, false, false, true}
			if diff := cmp.Diff(want, isPrime[:len(want)]); diff != "" {
				b.Errorf("miscalculated isPrime (-want,+got):\n%s", diff)
			}
		})
	}
}

func calPrime(p int) bool {
	if p <= 1 {
		return false
	}
	for i := 2; i*i <= p; i++ {
		if p%i == 0 {
			return false
		}
	}
	return true
}
