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
)

// MaxHashBits is the max bit length which can be used from hash value.
// If we use all bits of hash value, the critical(last) card shuffled by
// Dealer will be uneven to 2:3 (first half:second half) at most,
// in order to reduce this unevenness to 32:33, we set MaxHashBits to 60 here.
const MaxHashBits = 60

// RequiredEntropyBits makes a quick and slightly conservative estimate of the number
// of bits of hash value that are consumed in shuffle sharding a deck of the given size
// to a hand of the given size.  The result is meaningful only if
// 1 <= handSize <= deckSize <= 1<<26.
func RequiredEntropyBits(deckSize, handSize int) int {
	return int(math.Ceil(math.Log2(float64(deckSize)) * float64(handSize)))
}

// Dealer contains some necessary parameters and provides some methods for shuffle sharding.
// Dealer is thread-safe.
type Dealer struct {
	deckSize int
	handSize int
}

// NewDealer will create a Dealer with the given deckSize and handSize, will return error when
// deckSize or handSize is invalid as below.
// 1. deckSize or handSize is not positive
// 2. handSize is greater than deckSize
// 3. deckSize is impractically large (greater than 1<<26)
// 4. required entropy bits of deckSize and handSize is greater than MaxHashBits
func NewDealer(deckSize, handSize int) (*Dealer, error) {
	if deckSize <= 0 || handSize <= 0 {
		return nil, fmt.Errorf("deckSize %d or handSize %d is not positive", deckSize, handSize)
	}
	if handSize > deckSize {
		return nil, fmt.Errorf("handSize %d is greater than deckSize %d", handSize, deckSize)
	}
	if deckSize > 1<<26 {
		return nil, fmt.Errorf("deckSize %d is impractically large", deckSize)
	}
	if RequiredEntropyBits(deckSize, handSize) > MaxHashBits {
		return nil, fmt.Errorf("required entropy bits of deckSize %d and handSize %d is greater than %d", deckSize, handSize, MaxHashBits)
	}

	return &Dealer{
		deckSize: deckSize,
		handSize: handSize,
	}, nil
}

// Deal shuffles a card deck and deals a hand of cards, using the given hashValue as the source of entropy.
// The deck size and hand size are properties of the Dealer.
// This function synchronously makes sequential calls to pick, one for each dealt card.
// Each card is identified by an integer in the range [0, deckSize).
// For example, for deckSize=128 and handSize=4 this function might call pick(14); pick(73); pick(119); pick(26).
func (d *Dealer) Deal(hashValue uint64, pick func(int)) {
	// 15 is the largest possible value of handSize
	var remainders [15]int

	for i := 0; i < d.handSize; i++ {
		hashValueNext := hashValue / uint64(d.deckSize-i)
		remainders[i] = int(hashValue - uint64(d.deckSize-i)*hashValueNext)
		hashValue = hashValueNext
	}

	for i := 0; i < d.handSize; i++ {
		card := remainders[i]
		for j := i; j > 0; j-- {
			if card >= remainders[j-1] {
				card++
			}
		}
		pick(card)
	}
}

// DealIntoHand shuffles and deals according to the Dealer's parameters,
// using the given hashValue as the source of entropy and then
// returns the dealt cards as a slice of `int`.
// If `hand` has the correct length as Dealer's handSize, it will be used as-is and no allocations will be made.
// If `hand` is nil or too small, it will be extended (performing an allocation).
// If `hand` is too large, a sub-slice will be returned.
func (d *Dealer) DealIntoHand(hashValue uint64, hand []int) []int {
	h := hand[:0]
	d.Deal(hashValue, func(card int) { h = append(h, card) })
	return h
}
