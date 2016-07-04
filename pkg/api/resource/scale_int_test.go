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

package resource

import (
	"math"
	"math/big"
	"testing"
)

func TestScaledValueInternal(t *testing.T) {
	tests := []struct {
		unscaled *big.Int
		scale    int
		newScale int

		want int64
	}{
		// remain scale
		{big.NewInt(1000), 0, 0, 1000},

		// scale down
		{big.NewInt(1000), 0, -3, 1},
		{big.NewInt(1000), 3, 0, 1},
		{big.NewInt(0), 3, 0, 0},

		// always round up
		{big.NewInt(999), 3, 0, 1},
		{big.NewInt(500), 3, 0, 1},
		{big.NewInt(499), 3, 0, 1},
		{big.NewInt(1), 3, 0, 1},
		// large scaled value does not lose precision
		{big.NewInt(0).Sub(maxInt64, bigOne), 1, 0, (math.MaxInt64-1)/10 + 1},
		// large intermidiate result.
		{big.NewInt(1).Exp(big.NewInt(10), big.NewInt(100), nil), 100, 0, 1},

		// scale up
		{big.NewInt(0), 0, 3, 0},
		{big.NewInt(1), 0, 3, 1000},
		{big.NewInt(1), -3, 0, 1000},
		{big.NewInt(1000), -3, 2, 100000000},
		{big.NewInt(0).Div(big.NewInt(math.MaxInt64), bigThousand), 0, 3,
			(math.MaxInt64 / 1000) * 1000},
	}

	for i, tt := range tests {
		old := (&big.Int{}).Set(tt.unscaled)
		got := scaledValue(tt.unscaled, tt.scale, tt.newScale)
		if got != tt.want {
			t.Errorf("#%d: got = %v, want %v", i, got, tt.want)
		}
		if tt.unscaled.Cmp(old) != 0 {
			t.Errorf("#%d: unscaled = %v, want %v", i, tt.unscaled, old)
		}
	}
}

func BenchmarkScaledValueSmall(b *testing.B) {
	s := big.NewInt(1000)
	for i := 0; i < b.N; i++ {
		scaledValue(s, 3, 0)
	}
}

func BenchmarkScaledValueLarge(b *testing.B) {
	s := big.NewInt(math.MaxInt64)
	s.Mul(s, big.NewInt(1000))
	for i := 0; i < b.N; i++ {
		scaledValue(s, 10, 0)
	}
}
