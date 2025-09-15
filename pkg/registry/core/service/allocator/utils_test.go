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

func TestCountBits(t *testing.T) {
	// bigN is an integer that occupies more than one big.Word.
	bigN, ok := big.NewInt(0).SetString("10000000000000000000000000000000000000000000000000000000000000000", 16)
	if !ok {
		t.Fatal("Failed to set bigN")
	}
	tests := []struct {
		n        *big.Int
		expected int
	}{
		{n: big.NewInt(int64(0)), expected: 0},
		{n: big.NewInt(int64(0xffffffffff)), expected: 40},
		{n: bigN, expected: 1},
	}
	for _, test := range tests {
		actual := countBits(test.n)
		if test.expected != actual {
			t.Errorf("%s should have %d bits but recorded as %d", test.n, test.expected, actual)
		}
	}
}

func BenchmarkCountBits(b *testing.B) {
	bigN, ok := big.NewInt(0).SetString("10000000000000000000000000000000000000000000000000000000000000000", 16)
	if !ok {
		b.Fatal("Failed to set bigN")
	}
	for i := 0; i < b.N; i++ {
		countBits(bigN)
	}
}
