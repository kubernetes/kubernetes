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
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

func TestRequiredEntropyBits(t *testing.T) {
	tests := []struct {
		name         string
		deckSize     int
		handSize     int
		expectedBits int
	}{
		{
			"deckSize: 1024 handSize: 6",
			1024,
			6,
			60,
		},
		{
			"deckSize: 512 handSize: 8",
			512,
			8,
			72,
		},
	}

	for _, test := range tests {
		bits := RequiredEntropyBits(test.deckSize, test.handSize)
		if bits != test.expectedBits {
			t.Errorf("test %s fails: expected %v but got %v", test.name, test.expectedBits, bits)
			return
		}
	}
}

func TestNewDealer(t *testing.T) {
	tests := []struct {
		name     string
		deckSize int
		handSize int
		err      error
	}{
		{
			"deckSize <= 0",
			-100,
			8,
			fmt.Errorf("deckSize -100 or handSize 8 is not positive"),
		},
		{
			"handSize <= 0",
			100,
			0,
			fmt.Errorf("deckSize 100 or handSize 0 is not positive"),
		},
		{
			"handSize is greater than deckSize",
			100,
			101,
			fmt.Errorf("handSize 101 is greater than deckSize 100"),
		},
		{
			"deckSize is impractically large",
			1 << 27,
			2,
			fmt.Errorf("deckSize 134217728 is impractically large"),
		},
		{
			"required entropy bits is greater than MaxHashBits",
			512,
			8,
			fmt.Errorf("required entropy bits of deckSize 512 and handSize 8 is greater than 60"),
		},
		{
			"deckSize: 1024 handSize: 6",
			1024,
			6,
			nil,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewDealer(test.deckSize, test.handSize)
			if !reflect.DeepEqual(err, test.err) {
				t.Errorf("test %s fails: expected %v but got %v", test.name, test.err, err)
				return
			}
		})
	}
}

func TestCardDuplication(t *testing.T) {
	tests := []struct {
		name     string
		deckSize int
		handSize int
	}{
		{
			"deckSize = handSize = 4",
			4,
			4,
		},
		{
			"deckSize = handSize = 8",
			8,
			8,
		},
		{
			"deckSize = handSize = 10",
			10,
			10,
		},
		{
			"deckSize = handSize = 12",
			12,
			12,
		},
		{
			"deckSize = 128, handSize = 8",
			128,
			8,
		},
		{
			"deckSize = 256, handSize = 7",
			256,
			7,
		},
		{
			"deckSize = 512, handSize = 6",
			512,
			6,
		},
	}
	for _, test := range tests {
		hashValue := rand.Uint64()
		t.Run(test.name, func(t *testing.T) {
			dealer, err := NewDealer(test.deckSize, test.handSize)
			if err != nil {
				t.Errorf("fail to create Dealer: %v", err)
				return
			}
			hand := make([]int, 0)
			hand = dealer.DealIntoHand(hashValue, hand)

			// check cards number
			if len(hand) != int(test.handSize) {
				t.Errorf("test case %s fails in cards number", test.name)
				return
			}

			// check cards range and duplication
			cardMap := make(map[int]struct{})
			for _, card := range hand {
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

		})
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
	const spare = 64 - MaxHashBits
	tests := []struct {
		deckSize, handSize int
		hashMax            int
	}{
		{64, 3, 1 << uint(math.Ceil(math.Log2(float64(ff(64, 3))))+spare)},
		{128, 3, ff(128, 3)},
		{50, 4, ff(50, 4)},
	}
	for _, test := range tests {
		dealer, err := NewDealer(test.deckSize, test.handSize)
		if err != nil {
			t.Errorf("fail to create Dealer: %v", err)
			return
		}
		handCoordinateMap := make(map[int]int) // maps coded hand to count of times seen

		fallingFactorial := ff(test.deckSize, test.handSize)
		permutations := ff(test.handSize, test.handSize)
		allCoordinateCount := fallingFactorial / permutations
		nff := float64(test.hashMax) / float64(fallingFactorial)
		minCount := permutations * int(math.Floor(nff))
		maxCount := permutations * int(math.Ceil(nff))
		aHand := make([]int, test.handSize)
		for i := 0; i < test.hashMax; i++ {
			aHand = dealer.DealIntoHand(uint64(i), aHand)
			sort.IntSlice(aHand).Sort()
			handCoordinate := 0
			for _, card := range aHand {
				handCoordinate = handCoordinate<<7 + card
			}
			handCoordinateMap[handCoordinate]++
		}
		numHandsSeen := len(handCoordinateMap)

		t.Logf("Deck size = %v, hand size = %v, number of possible hands = %d, number of hands seen = %d, number of deals = %d, expected count range = [%v, %v]", test.deckSize, test.handSize, allCoordinateCount, numHandsSeen, test.hashMax, minCount, maxCount)

		// histogram maps (count of times a hand is seen) to (number of hands having that count)
		histogram := make(map[int]int)
		for _, count := range handCoordinateMap {
			histogram[count] = histogram[count] + 1
		}

		var goodSum int
		for count := minCount; count <= maxCount; count++ {
			goodSum += histogram[count]
		}

		goodPct := 100 * float64(goodSum) / float64(numHandsSeen)

		t.Logf("good percentage = %v, histogram = %v", goodPct, histogram)
		if goodSum != numHandsSeen {
			t.Errorf("Only %v percent of the hands got a central count", goodPct)
		}
	}
}

func TestDealer_DealIntoHand(t *testing.T) {
	dealer, _ := NewDealer(6, 6)

	tests := []struct {
		name         string
		hand         []int
		expectedSize int
	}{
		{
			"nil slice",
			nil,
			6,
		},
		{
			"empty slice",
			make([]int, 0),
			6,
		},
		{
			"size: 6 cap: 6 slice",
			make([]int, 6),
			6,
		},
		{
			"size: 6 cap: 12 slice",
			make([]int, 6, 12),
			6,
		},
		{
			"size: 4 cap: 4 slice",
			make([]int, 4),
			6,
		},
		{
			"size: 4 cap: 12 slice",
			make([]int, 4, 12),
			6,
		},
		{
			"size: 10 cap: 10 slice",
			make([]int, 10),
			6,
		},
		{
			"size: 10 cap: 12 slice",
			make([]int, 10, 12),
			6,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			h := dealer.DealIntoHand(0, test.hand)
			if len(h) != test.expectedSize {
				t.Errorf("test %s fails: expetced size %d but got %d", test.name, test.expectedSize, len(h))
				return
			}
		})
	}
}
