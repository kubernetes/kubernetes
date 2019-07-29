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
	"math"
)

const maxHashBits = 60

// ValidateParameters can validate parameters for shuffle sharding
// in a fast but approximate way, including numQueues and handSize
// Algorithm: maxHashValue >= numQueues^handSize
func ValidateParameters(numQueues, handSize int32) bool {
	if handSize <= 0 || numQueues <= 0 || handSize > numQueues {
		return false
	}

	return math.Log2(float64(numQueues))*float64(handSize) <= maxHashBits
}

// Deal can shuffle a hash value to handSize-quantity and non-redundant
// indices of queue, with the pick function, we can get the optimal queue index
// Eg. From numQueues=128, handSize=8, we can get an index array [12 14 73 18 119 51 117 26],
// then pick function will choose the optimal index from these
// Algorithm: https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190228-priority-and-fairness.md#queue-assignment-proof-of-concept
func Deal(hashValue uint64, numQueues, handSize int32, pick func(int32) error) error {
	remainders := make([]int32, handSize)

	for i := int32(0); i < handSize; i++ {
		hashValueNext := hashValue / uint64(numQueues-i)
		remainders[i] = int32(hashValue - uint64(numQueues-i)*hashValueNext)
		hashValue = hashValueNext
	}

	for i := int32(0); i < handSize; i++ {
		candidate := remainders[i]
		for j := i; j > 0; j-- {
			if candidate >= remainders[j-1] {
				candidate++
			}
		}
		if err := pick(candidate); err != nil {
			return err
		}
	}

	return nil
}

// DealWithValidation will do validation before Deal
func DealWithValidation(hashValue uint64, numQueues, handSize int32, pick func(int32) error) error {
	if !ValidateParameters(numQueues, handSize) {
		return errors.New("bad parameters")
	}

	return Deal(hashValue, numQueues, handSize, pick)
}

// DealToSlices will use specific pick function to return slices of indices
// after Deal
func DealToSlices(hashValue uint64, numQueues, handSize int32) ([]int32, error) {
	if !ValidateParameters(numQueues, handSize) {
		return nil, errors.New("bad parameters")
	}

	var (
		candidates = make([]int32, handSize)
		idx        = 0
	)

	pickToSlices := func(can int32) error {
		candidates[idx] = can
		idx++
		return nil
	}

	if err := Deal(hashValue, numQueues, handSize, pickToSlices); err != nil {
		return nil, err
	}

	return candidates, nil
}
