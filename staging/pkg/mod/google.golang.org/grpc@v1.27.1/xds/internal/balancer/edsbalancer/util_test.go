/*
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package edsbalancer

import (
	"sync"
	"testing"

	"google.golang.org/grpc/internal/wrr"
)

// testWRR is a deterministic WRR implementation.
//
// The real implementation does random WRR. testWRR makes the balancer behavior
// deterministic and easier to test.
//
// With {a: 2, b: 3}, the Next() results will be {a, a, b, b, b}.
type testWRR struct {
	itemsWithWeight []struct {
		item   interface{}
		weight int64
	}
	length int

	mu    sync.Mutex
	idx   int   // The index of the item that will be picked
	count int64 // The number of times the current item has been picked.
}

func newTestWRR() wrr.WRR {
	return &testWRR{}
}

func (twrr *testWRR) Add(item interface{}, weight int64) {
	twrr.itemsWithWeight = append(twrr.itemsWithWeight, struct {
		item   interface{}
		weight int64
	}{item: item, weight: weight})
	twrr.length++
}

func (twrr *testWRR) Next() interface{} {
	twrr.mu.Lock()
	iww := twrr.itemsWithWeight[twrr.idx]
	twrr.count++
	if twrr.count >= iww.weight {
		twrr.idx = (twrr.idx + 1) % twrr.length
		twrr.count = 0
	}
	twrr.mu.Unlock()
	return iww.item
}

func init() {
	newRandomWRR = newTestWRR
}

func TestDropper(t *testing.T) {
	const repeat = 2

	type args struct {
		numerator   uint32
		denominator uint32
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "2_3",
			args: args{
				numerator:   2,
				denominator: 3,
			},
		},
		{
			name: "4_8",
			args: args{
				numerator:   4,
				denominator: 8,
			},
		},
		{
			name: "7_20",
			args: args{
				numerator:   7,
				denominator: 20,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := newDropper(tt.args.numerator, tt.args.denominator, "")
			var (
				dCount    int
				wantCount = int(tt.args.numerator) * repeat
				loopCount = int(tt.args.denominator) * repeat
			)
			for i := 0; i < loopCount; i++ {
				if d.drop() {
					dCount++
				}
			}

			if dCount != (wantCount) {
				t.Errorf("with numerator %v, denominator %v repeat %v, got drop count: %v, want %v",
					tt.args.numerator, tt.args.denominator, repeat, dCount, wantCount)
			}
		})
	}
}
