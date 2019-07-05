/*
Copyright 2015 The Kubernetes Authors.

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

package allocator

import (
	"math/big"
	"testing"
)

func TestBitCount(t *testing.T) {
	for i, c := range bitCounts {
		actual := 0
		for j := 0; j < 8; j++ {
			if ((1 << uint(j)) & i) != 0 {
				actual++
			}
		}
		if actual != int(c) {
			t.Errorf("%d should have %d bits but recorded as %d", i, actual, c)
		}
	}
}

func TestCountBits(t *testing.T) {
	tests := []struct {
		n        *big.Int
		expected int
	}{
		{n: big.NewInt(int64(0)), expected: 0},
		{n: big.NewInt(int64(0xffffffffff)), expected: 40},
	}
	for _, test := range tests {
		actual := countBits(test.n)
		if test.expected != actual {
			t.Errorf("%s should have %d bits but recorded as %d", test.n, test.expected, actual)
		}
	}
}
