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
	"errors"
	"fmt"
	"math"
	"strings"
)

const maxHashBits = 60

// ValidateParameters finds errors in the parameters for shuffle
// sharding.  Returns a slice for which `len()` is 0 if and only if
// there are no errors.  The entropy requirement is evaluated in a
// fast but approximate way: bits(deckSize^handSize).
func ValidateParameters(deckSize, handSize int) (errs []string) {
	if handSize <= 0 {
		errs = append(errs, "handSize is not positive")
	}
	if deckSize <= 0 {
		errs = append(errs, "deckSize is not positive")
	}
	if len(errs) > 0 {
		return
	}
	if handSize > deckSize {
		return []string{"handSize is greater than deckSize"}
	}
	if math.Log2(float64(deckSize))*float64(handSize) > maxHashBits {
		return []string{fmt.Sprintf("more than %d bits of entropy required", maxHashBits)}
	}
	return
}

// ShuffleAndDeal can shuffle a hash value to handSize-quantity and non-redundant
// indices of decks, with the pick function, we can get the optimal deck index
// Eg. From deckSize=128, handSize=8, we can get an index array [12 14 73 18 119 51 117 26],
// then pick function will choose the optimal index from these
// Algorithm: https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190228-priority-and-fairness.md#queue-assignment-proof-of-concept
func ShuffleAndDeal(hashValue uint64, deckSize, handSize int, pick func(int)) {
	remainders := make([]int, handSize)

	for i := 0; i < handSize; i++ {
		hashValueNext := hashValue / uint64(deckSize-i)
		remainders[i] = int(hashValue - uint64(deckSize-i)*hashValueNext)
		hashValue = hashValueNext
	}

	for i := 0; i < handSize; i++ {
		candidate := remainders[i]
		for j := i; j > 0; j-- {
			if candidate >= remainders[j-1] {
				candidate++
			}
		}
		pick(candidate)
	}
}

// ShuffleAndDealWithValidation will do validation before ShuffleAndDeal
func ShuffleAndDealWithValidation(hashValue uint64, deckSize, handSize int, pick func(int)) error {
	if errs := ValidateParameters(deckSize, handSize); len(errs) > 0 {
		return errors.New(strings.Join(errs, ";"))
	}

	ShuffleAndDeal(hashValue, deckSize, handSize, pick)
	return nil
}

// ShuffleAndDealToSlice will use specific pick function to return slices of indices
// after ShuffleAndDeal
func ShuffleAndDealToSlice(hashValue uint64, deckSize, handSize int) []int {
	var (
		candidates = make([]int, handSize)
		idx        = 0
	)

	pickToSlices := func(can int) {
		candidates[idx] = int(can)
		idx++
	}

	ShuffleAndDeal(hashValue, deckSize, handSize, pickToSlices)

	return candidates
}

// ShuffleAndDealIntoHand shuffles a deck of the given size by the
// given hash value and deals cards into the given slice.  The virtue
// of this function compared to ShuffleAndDealToSlice is that the
// caller provides the storage for the hand.
func ShuffleAndDealIntoHand(hashValue uint64, deckSize int, hand []int) {
	handSize := len(hand)
	var idx int
	ShuffleAndDeal(hashValue, deckSize, handSize, func(card int) {
		hand[idx] = int(card)
		idx++
	})
}

// ShuffleAndDealToSliceWithValidation will do validation before ShuffleAndDealToSlice
func ShuffleAndDealToSliceWithValidation(hashValue uint64, deckSize, handSize int) ([]int, error) {
	if errs := ValidateParameters(deckSize, handSize); len(errs) > 0 {
		return nil, errors.New(strings.Join(errs, ";"))
	}

	return ShuffleAndDealToSlice(hashValue, deckSize, handSize), nil
}

// ShuffleAndDealIntoHandWithValidation does validation and then ShuffleAndDealIntoHand
func ShuffleAndDealIntoHandWithValidation(hashValue uint64, deckSize int, hand []int) error {
	if errs := ValidateParameters(deckSize, len(hand)); len(errs) > 0 {
		return errors.New(strings.Join(errs, ";"))
	}
	ShuffleAndDealIntoHand(hashValue, deckSize, hand)
	return nil
}
