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

package shufflesharding

import (
	"math"
	"math/rand"
	"sort"
	"strings"
	"testing"
)

func TestValidateParameters(t *testing.T) {
	tests := []struct {
		name     string
		deckSize int
		handSize int
		errors   []string
	}{
		{
			"deckSize is < 0",
			-100,
			8,
			[]string{"deckSize is not positive"},
		},
		{
			"handSize is < 0",
			128,
			-100,
			[]string{"handSize is not positive"},
		},
		{
			"deckSize is 0",
			0,
			8,
			[]string{"deckSize is not positive"},
		},
		{
			"handSize is 0",
			128,
			0,
			[]string{"handSize is not positive"},
		},
		{
			"handSize is greater than deckSize",
			128,
			129,
			[]string{"handSize is greater than deckSize"},
		},
		{
			"deckSize: 128 handSize: 6",
			128,
			6,
			nil,
		},
		{
			"deckSize: 1024 handSize: 6",
			1024,
			6,
			nil,
		},
		{
			"deckSize: 512 handSize: 8",
			512,
			8,
			[]string{"more than 60 bits of entropy required"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := strings.Join(ValidateParameters(test.deckSize, test.handSize), ";")
			expected := strings.Join(test.errors, ";")
			if got != expected {
				t.Errorf("test case %s got %q but expected %q", test.name, got, expected)
				return
			}
		})
	}
}

func BenchmarkValidateParameters(b *testing.B) {
	deckSize, handSize := 512, 8
	for i := 0; i < b.N; i++ {
		_ = ValidateParameters(deckSize, handSize)
	}
}

func TestShuffleAndDealWithValidation(t *testing.T) {
	tests := []struct {
		name      string
		deckSize  int
		handSize  int
		pick      func(int)
		validated bool
	}{
		{
			"deckSize is < 0",
			-100,
			8,
			func(int) {},
			false,
		},
		{
			"handSize is < 0",
			128,
			-100,
			func(int) {},
			false,
		},
		{
			"deckSize is 0",
			0,
			8,
			func(int) {},
			false,
		},
		{
			"handSize is 0",
			128,
			0,
			func(int) {},
			false,
		},
		{
			"handSize is greater than deckSize",
			128,
			129,
			func(int) {},
			false,
		},
		{
			"deckSize: 128 handSize: 6",
			128,
			6,
			func(int) {},
			true,
		},
		{
			"deckSize: 1024 handSize: 6",
			1024,
			6,
			func(int) {},
			true,
		},
		{
			"deckSize: 512 handSize: 8",
			512,
			8,
			func(int) {},
			false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if (ShuffleAndDealWithValidation(rand.Uint64(), test.deckSize, test.handSize, test.pick) == nil) != test.validated {
				t.Errorf("test case %s fails", test.name)
				return
			}
		})
	}
}

func BenchmarkShuffleAndDeal(b *testing.B) {
	hashValueBase := math.MaxUint64 / uint64(b.N)
	deckSize, handSize := 512, 8
	pick := func(int) {}
	for i := 0; i < b.N; i++ {
		ShuffleAndDeal(hashValueBase*uint64(i), deckSize, handSize, pick)
	}
}

func TestShuffleAndDealToSliceWithValidation(t *testing.T) {
	tests := []struct {
		name      string
		deckSize  int
		handSize  int
		validated bool
	}{
		{
			"validation fails",
			-100,
			-100,
			false,
		},
		{
			"deckSize = handSize = 4",
			4,
			4,
			true,
		},
		{
			"deckSize = handSize = 8",
			8,
			8,
			true,
		},
		{
			"deckSize = handSize = 10",
			10,
			10,
			true,
		},
		{
			"deckSize = handSize = 12",
			12,
			12,
			true,
		},
		{
			"deckSize = 128, handSize = 8",
			128,
			8,
			true,
		},
		{
			"deckSize = 256, handSize = 7",
			256,
			7,
			true,
		},
		{
			"deckSize = 512, handSize = 6",
			512,
			6,
			true,
		},
	}
	for _, test := range tests {
		hashValue := rand.Uint64()
		t.Run(test.name, func(t *testing.T) {
			cards, err := ShuffleAndDealToSliceWithValidation(hashValue, test.deckSize, test.handSize)
			if (err == nil) != test.validated {
				t.Errorf("test case %s fails in validation check", test.name)
				return
			}

			if test.validated {
				// check cards number
				if len(cards) != int(test.handSize) {
					t.Errorf("test case %s fails in cards number", test.name)
					return
				}

				// check cards range and duplication
				cardMap := make(map[int]struct{})
				for _, card := range cards {
					if card < 0 || card >= int(test.deckSize) {
						t.Errorf("test case %s fails in range check", test.name)
						return
					}
					cardMap[card] = struct{}{}
				}
				if len(cardMap) != int(test.handSize) {
					t.Errorf("test case %s fails in duplication check", test.name)
					return
				}
			}
		})
	}
}

func BenchmarkShuffleAndDealToSlice(b *testing.B) {
	hashValueBase := math.MaxUint64 / uint64(b.N)
	deckSize, handSize := 512, 8
	for i := 0; i < b.N; i++ {
		_ = ShuffleAndDealToSlice(hashValueBase*uint64(i), deckSize, handSize)
	}
}

// ff computes the falling factorial `n!/(n-m)!` and requires n to be
// positive and m to be in the range [0, n] and requires the answer to
// fit in an int
func ff(n, m int) int {
	ans := 1
	for f := n; f > n-m; f-- {
		ans *= f
	}
	return ans
}

func TestUniformDistribution(t *testing.T) {
	const spare = 64 - maxHashBits
	tests := []struct {
		deckSize, handSize int
		hashMax            int
	}{
		{64, 3, 1 << uint(math.Ceil(math.Log2(float64(ff(64, 3))))+spare)},
		{128, 3, ff(128, 3)},
		{128, 3, 3 * ff(128, 3)},
		{70, 4, ff(70, 4)},
	}
	for _, test := range tests {
		handCoordinateMap := make(map[int]int) // maps coded hand to count of times seen

		fallingFactorial := ff(test.deckSize, test.handSize)
		permutations := ff(test.handSize, test.handSize)
		allCoordinateCount := fallingFactorial / permutations
		nff := float64(test.hashMax) / float64(fallingFactorial)
		minCount := permutations * int(math.Floor(nff))
		maxCount := permutations * int(math.Ceil(nff))
		aHand := make([]int, test.handSize)
		for i := 0; i < test.hashMax; i++ {
			ShuffleAndDealIntoHand(uint64(i), test.deckSize, aHand)
			sort.IntSlice(aHand).Sort()
			handCoordinate := 0
			for _, card := range aHand {
				handCoordinate = handCoordinate<<7 + card
			}
			handCoordinateMap[handCoordinate]++
		}

		t.Logf("Deck size = %v, hand size = %v, number of possible hands = %d, number of hands seen = %d, number of deals = %d, expected count range = [%v, %v]", test.deckSize, test.handSize, allCoordinateCount, len(handCoordinateMap), test.hashMax, minCount, maxCount)

		// histogram maps (count of times a hand is seen) to (number of hands having that count)
		histogram := make(map[int]int)
		for _, count := range handCoordinateMap {
			histogram[count] = histogram[count] + 1
		}

		var goodSum int
		for count := minCount; count <= maxCount; count++ {
			goodSum += histogram[count]
		}

		goodPct := 100 * float64(goodSum) / float64(allCoordinateCount)

		t.Logf("good percentage = %v, histogram = %v", goodPct, histogram)
		if goodSum != allCoordinateCount {
			t.Errorf("Only %v percent of the hands got a central count", goodPct)
		}
	}
	return
}
