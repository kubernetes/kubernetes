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

package bitmask

import (
	"reflect"
	"testing"
)

func TestNewEmptyiBitMask(t *testing.T) {
	tcases := []struct {
		name         string
		expectedMask string
	}{
		{
			name:         "New empty BitMask",
			expectedMask: "00",
		},
	}
	for _, tc := range tcases {
		bm := NewEmptyBitMask()
		if bm.String() != tc.expectedMask {
			t.Errorf("Expected mask to be %v, got %v", tc.expectedMask, bm)
		}
	}
}

func TestNewBitMask(t *testing.T) {
	tcases := []struct {
		name         string
		bits         []int
		expectedMask string
	}{
		{
			name:         "New BitMask with bit 0 set",
			bits:         []int{0},
			expectedMask: "01",
		},
		{
			name:         "New BitMask with bit 1 set",
			bits:         []int{1},
			expectedMask: "10",
		},
		{
			name:         "New BitMask with bit 0 and bit 1 set",
			bits:         []int{0, 1},
			expectedMask: "11",
		},
	}
	for _, tc := range tcases {
		mask, _ := NewBitMask(tc.bits...)
		if mask.String() != tc.expectedMask {
			t.Errorf("Expected mask to be %v, got %v", tc.expectedMask, mask)
		}
	}
}

func TestAdd(t *testing.T) {
	tcases := []struct {
		name         string
		bits         []int
		expectedMask string
	}{
		{
			name:         "Add BitMask with bit 0 set",
			bits:         []int{0},
			expectedMask: "01",
		},
		{
			name:         "Add BitMask with bit 1 set",
			bits:         []int{1},
			expectedMask: "10",
		},
		{
			name:         "Add BitMask with bits 0 and 1 set",
			bits:         []int{0, 1},
			expectedMask: "11",
		},
		{
			name:         "Add BitMask with bits outside range 0-63",
			bits:         []int{-1, 64},
			expectedMask: "00",
		},
	}
	for _, tc := range tcases {
		mask, _ := NewBitMask()
		mask.Add(tc.bits...)
		if mask.String() != tc.expectedMask {
			t.Errorf("Expected mask to be %v, got %v", tc.expectedMask, mask)
		}
	}
}

func TestRemove(t *testing.T) {
	tcases := []struct {
		name         string
		bitsSet      []int
		bitsRemove   []int
		expectedMask string
	}{
		{
			name:         "Set bit 0. Remove bit 0",
			bitsSet:      []int{0},
			bitsRemove:   []int{0},
			expectedMask: "00",
		},
		{
			name:         "Set bits 0 and 1. Remove bit 1",
			bitsSet:      []int{0, 1},
			bitsRemove:   []int{1},
			expectedMask: "01",
		},
		{
			name:         "Set bits 0 and 1. Remove bits 0 and 1",
			bitsSet:      []int{0, 1},
			bitsRemove:   []int{0, 1},
			expectedMask: "00",
		},
		{
			name:         "Set bit 0. Attempt to remove bits outside range 0-63",
			bitsSet:      []int{0},
			bitsRemove:   []int{-1, 64},
			expectedMask: "01",
		},
	}
	for _, tc := range tcases {
		mask, _ := NewBitMask(tc.bitsSet...)
		mask.Remove(tc.bitsRemove...)
		if mask.String() != tc.expectedMask {
			t.Errorf("Expected mask to be %v, got %v", tc.expectedMask, mask)
		}
	}
}

func TestAnd(t *testing.T) {
	tcases := []struct {
		name    string
		masks   [][]int
		andMask string
	}{
		{
			name:    "Mask 11 AND mask 11",
			masks:   [][]int{{0, 1}, {0, 1}},
			andMask: "11",
		},
		{
			name:    "Mask 11 AND mask 10",
			masks:   [][]int{{0, 1}, {1}},
			andMask: "10",
		},
		{
			name:    "Mask 01 AND mask 11",
			masks:   [][]int{{0}, {0, 1}},
			andMask: "01",
		},
		{
			name:    "Mask 11 AND mask 11 AND mask 10",
			masks:   [][]int{{0, 1}, {0, 1}, {1}},
			andMask: "10",
		},
		{
			name:    "Mask 01 AND mask 01 AND mask 10 AND mask 11",
			masks:   [][]int{{0}, {0}, {1}, {0, 1}},
			andMask: "00",
		},
		{
			name:    "Mask 1111 AND mask 1110 AND mask 1100 AND mask 1000",
			masks:   [][]int{{0, 1, 2, 3}, {1, 2, 3}, {2, 3}, {3}},
			andMask: "1000",
		},
	}
	for _, tc := range tcases {
		var bitMasks []BitMask
		for i := range tc.masks {
			bitMask, _ := NewBitMask(tc.masks[i]...)
			bitMasks = append(bitMasks, bitMask)
		}
		resultMask := And(bitMasks[0], bitMasks...)
		if resultMask.String() != string(tc.andMask) {
			t.Errorf("Expected mask to be %v, got %v", tc.andMask, resultMask)
		}

	}
}

func TestOr(t *testing.T) {
	tcases := []struct {
		name   string
		masks  [][]int
		orMask string
	}{
		{
			name:   "Mask 01 OR mask 00",
			masks:  [][]int{{0}, {}},
			orMask: "01",
		},
		{
			name:   "Mask 10 OR mask 10",
			masks:  [][]int{{1}, {1}},
			orMask: "10",
		},
		{
			name:   "Mask 01 OR mask 10",
			masks:  [][]int{{0}, {1}},
			orMask: "11",
		},
		{
			name:   "Mask 11 OR mask 11",
			masks:  [][]int{{0, 1}, {0, 1}},
			orMask: "11",
		},
		{
			name:   "Mask 01 OR mask 10 OR mask 11",
			masks:  [][]int{{0}, {1}, {0, 1}},
			orMask: "11",
		},
		{
			name:   "Mask 1000 OR mask 0100 OR mask 0010 OR mask 0001",
			masks:  [][]int{{3}, {2}, {1}, {0}},
			orMask: "1111",
		},
	}
	for _, tc := range tcases {
		var bitMasks []BitMask
		for i := range tc.masks {
			bitMask, _ := NewBitMask(tc.masks[i]...)
			bitMasks = append(bitMasks, bitMask)
		}
		resultMask := Or(bitMasks[0], bitMasks...)
		if resultMask.String() != string(tc.orMask) {
			t.Errorf("Expected mask to be %v, got %v", tc.orMask, resultMask)
		}
	}
}

func TestClear(t *testing.T) {
	tcases := []struct {
		name        string
		mask        []int
		clearedMask string
	}{
		{
			name:        "Clear mask 01",
			mask:        []int{0},
			clearedMask: "00",
		},
		{
			name:        "Clear mask 10",
			mask:        []int{1},
			clearedMask: "00",
		},
		{
			name:        "Clear mask 11",
			mask:        []int{0, 1},
			clearedMask: "00",
		},
	}
	for _, tc := range tcases {
		mask, _ := NewBitMask(tc.mask...)
		mask.Clear()
		if mask.String() != string(tc.clearedMask) {
			t.Errorf("Expected mask to be %v, got %v", tc.clearedMask, mask)
		}
	}
}

func TestFill(t *testing.T) {
	tcases := []struct {
		name       string
		mask       []int
		filledMask string
	}{
		{
			name:       "Fill empty mask",
			mask:       nil,
			filledMask: "1111111111111111111111111111111111111111111111111111111111111111",
		},
		{
			name:       "Fill mask 10",
			mask:       []int{0},
			filledMask: "1111111111111111111111111111111111111111111111111111111111111111",
		},
		{
			name:       "Fill mask 11",
			mask:       []int{0, 1},
			filledMask: "1111111111111111111111111111111111111111111111111111111111111111",
		},
	}
	for _, tc := range tcases {
		mask, _ := NewBitMask(tc.mask...)
		mask.Fill()
		if mask.String() != string(tc.filledMask) {
			t.Errorf("Expected mask to be %v, got %v", tc.filledMask, mask)
		}
	}
}

func TestIsEmpty(t *testing.T) {
	tcases := []struct {
		name          string
		mask          []int
		expectedEmpty bool
	}{
		{
			name:          "Check if mask 00 is empty",
			mask:          nil,
			expectedEmpty: true,
		},
		{
			name:          "Check if mask 01 is empty",
			mask:          []int{0},
			expectedEmpty: false,
		},
		{
			name:          "Check if mask 11 is empty",
			mask:          []int{0, 1},
			expectedEmpty: false,
		},
	}
	for _, tc := range tcases {
		mask, _ := NewBitMask(tc.mask...)
		empty := mask.IsEmpty()
		if empty != tc.expectedEmpty {
			t.Errorf("Expected value to be %v, got %v", tc.expectedEmpty, empty)
		}
	}
}

func TestIsSet(t *testing.T) {
	tcases := []struct {
		name        string
		mask        []int
		checkBit    int
		expectedSet bool
	}{
		{
			name:        "Check if bit 0 in mask 00 is set",
			mask:        nil,
			checkBit:    0,
			expectedSet: false,
		},
		{
			name:        "Check if bit 0 in mask 01 is set",
			mask:        []int{0},
			checkBit:    0,
			expectedSet: true,
		},
		{
			name:        "Check if bit 1 in mask 11 is set",
			mask:        []int{0, 1},
			checkBit:    1,
			expectedSet: true,
		},
		{
			name:        "Check if bit outside range 0-63 is set",
			mask:        []int{0, 1},
			checkBit:    64,
			expectedSet: false,
		},
	}
	for _, tc := range tcases {
		mask, _ := NewBitMask(tc.mask...)
		set := mask.IsSet(tc.checkBit)
		if set != tc.expectedSet {
			t.Errorf("Expected value to be %v, got %v", tc.expectedSet, set)
		}
	}
}

func TestAnySet(t *testing.T) {
	tcases := []struct {
		name        string
		mask        []int
		checkBits   []int
		expectedSet bool
	}{
		{
			name:        "Check if any bits from 11 in mask 00 is set",
			mask:        nil,
			checkBits:   []int{0, 1},
			expectedSet: false,
		},
		{
			name:        "Check if any bits from 11 in mask 01 is set",
			mask:        []int{0},
			checkBits:   []int{0, 1},
			expectedSet: true,
		},
		{
			name:        "Check if any bits from 11 in mask 11 is set",
			mask:        []int{0, 1},
			checkBits:   []int{0, 1},
			expectedSet: true,
		},
		{
			name:        "Check if any bit outside range 0-63 is set",
			mask:        []int{0, 1},
			checkBits:   []int{64, 65},
			expectedSet: false,
		},
		{
			name:        "Check if any bits from 1001 in mask 0110 is set",
			mask:        []int{1, 2},
			checkBits:   []int{0, 3},
			expectedSet: false,
		},
	}
	for _, tc := range tcases {
		mask, _ := NewBitMask(tc.mask...)
		set := mask.AnySet(tc.checkBits)
		if set != tc.expectedSet {
			t.Errorf("Expected value to be %v, got %v", tc.expectedSet, set)
		}
	}
}

func TestIsEqual(t *testing.T) {
	tcases := []struct {
		name          string
		firstMask     []int
		secondMask    []int
		expectedEqual bool
	}{
		{
			name:          "Check if mask 00 equals mask 00",
			firstMask:     nil,
			secondMask:    nil,
			expectedEqual: true,
		},
		{
			name:          "Check if mask 00 equals mask 01",
			firstMask:     nil,
			secondMask:    []int{0},
			expectedEqual: false,
		},
		{
			name:          "Check if mask 01 equals mask 01",
			firstMask:     []int{0},
			secondMask:    []int{0},
			expectedEqual: true,
		},
		{
			name:          "Check if mask 01 equals mask 10",
			firstMask:     []int{0},
			secondMask:    []int{1},
			expectedEqual: false,
		},
		{
			name:          "Check if mask 11 equals mask 11",
			firstMask:     []int{0, 1},
			secondMask:    []int{0, 1},
			expectedEqual: true,
		},
	}
	for _, tc := range tcases {
		firstMask, _ := NewBitMask(tc.firstMask...)
		secondMask, _ := NewBitMask(tc.secondMask...)
		isEqual := firstMask.IsEqual(secondMask)
		if isEqual != tc.expectedEqual {
			t.Errorf("Expected mask to be %v, got %v", tc.expectedEqual, isEqual)
		}
	}
}

func TestCount(t *testing.T) {
	tcases := []struct {
		name          string
		bits          []int
		expectedCount int
	}{
		{
			name:          "Count number of bits set in mask 00",
			bits:          nil,
			expectedCount: 0,
		},
		{
			name:          "Count number of bits set in mask 01",
			bits:          []int{0},
			expectedCount: 1,
		},
		{
			name:          "Count number of bits set in mask 11",
			bits:          []int{0, 1},
			expectedCount: 2,
		},
	}
	for _, tc := range tcases {
		mask, _ := NewBitMask(tc.bits...)
		count := mask.Count()
		if count != tc.expectedCount {
			t.Errorf("Expected value to be %v, got %v", tc.expectedCount, count)
		}
	}
}

func TestGetBits(t *testing.T) {
	tcases := []struct {
		name         string
		bits         []int
		expectedBits []int
	}{
		{
			name:         "Get bits of mask 00",
			bits:         nil,
			expectedBits: nil,
		},
		{
			name:         "Get bits of mask 01",
			bits:         []int{0},
			expectedBits: []int{0},
		},
		{
			name:         "Get bits of mask 11",
			bits:         []int{0, 1},
			expectedBits: []int{0, 1},
		},
	}
	for _, tc := range tcases {
		mask, _ := NewBitMask(tc.bits...)
		bits := mask.GetBits()
		if !reflect.DeepEqual(bits, tc.expectedBits) {
			t.Errorf("Expected value to be %v, got %v", tc.expectedBits, bits)
		}
	}
}

func TestIsNarrowerThan(t *testing.T) {
	tcases := []struct {
		name                  string
		firstMask             []int
		secondMask            []int
		expectedFirstNarrower bool
	}{
		{
			name:                  "Check narrowness of masks with unequal bits set 1/2",
			firstMask:             []int{0},
			secondMask:            []int{0, 1},
			expectedFirstNarrower: true,
		},
		{
			name:                  "Check narrowness of masks with unequal bits set 2/2",
			firstMask:             []int{0, 1},
			secondMask:            []int{0},
			expectedFirstNarrower: false,
		},
		{
			name:                  "Check narrowness of masks with equal bits set 1/2",
			firstMask:             []int{0},
			secondMask:            []int{1},
			expectedFirstNarrower: true,
		},
		{
			name:                  "Check narrowness of masks with equal bits set 2/2",
			firstMask:             []int{1},
			secondMask:            []int{0},
			expectedFirstNarrower: false,
		},
	}
	for _, tc := range tcases {
		firstMask, _ := NewBitMask(tc.firstMask...)
		secondMask, _ := NewBitMask(tc.secondMask...)
		expectedFirstNarrower := firstMask.IsNarrowerThan(secondMask)
		if expectedFirstNarrower != tc.expectedFirstNarrower {
			t.Errorf("Expected value to be %v, got %v", tc.expectedFirstNarrower, expectedFirstNarrower)
		}
	}
}

func TestIterateBitMasks(t *testing.T) {
	tcases := []struct {
		name    string
		numbits int
	}{
		{
			name:    "1 bit",
			numbits: 1,
		},
		{
			name:    "2 bits",
			numbits: 2,
		},
		{
			name:    "4 bits",
			numbits: 4,
		},
		{
			name:    "8 bits",
			numbits: 8,
		},
		{
			name:    "16 bits",
			numbits: 16,
		},
	}
	for _, tc := range tcases {
		// Generate a list of bits from tc.numbits.
		var bits []int
		for i := 0; i < tc.numbits; i++ {
			bits = append(bits, i)
		}

		// Calculate the expected number of masks. Since we always have masks
		// with bits from 0..n, this is just (2^n - 1) since we want 1 mask
		// represented by each integer between 1 and 2^n-1.
		expectedNumMasks := (1 << uint(tc.numbits)) - 1

		// Iterate all masks and count them.
		numMasks := 0
		IterateBitMasks(bits, func(BitMask) {
			numMasks++
		})

		// Compare the number of masks generated to the expected amount.
		if expectedNumMasks != numMasks {
			t.Errorf("Expected to iterate %v masks, got %v", expectedNumMasks, numMasks)
		}
	}
}
