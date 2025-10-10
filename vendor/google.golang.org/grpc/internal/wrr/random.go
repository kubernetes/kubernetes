/*
 *
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

package wrr

import (
	"fmt"
	rand "math/rand/v2"
	"sort"
)

// weightedItem is a wrapped weighted item that is used to implement weighted random algorithm.
type weightedItem struct {
	item              any
	weight            int64
	accumulatedWeight int64
}

func (w *weightedItem) String() string {
	return fmt.Sprint(*w)
}

// randomWRR is a struct that contains weighted items implement weighted random algorithm.
type randomWRR struct {
	items []*weightedItem
	// Are all item's weights equal
	equalWeights bool
}

// NewRandom creates a new WRR with random.
func NewRandom() WRR {
	return &randomWRR{}
}

var randInt64n = rand.Int64N

func (rw *randomWRR) Next() (item any) {
	if len(rw.items) == 0 {
		return nil
	}
	if rw.equalWeights {
		return rw.items[randInt64n(int64(len(rw.items)))].item
	}

	sumOfWeights := rw.items[len(rw.items)-1].accumulatedWeight
	// Random number in [0, sumOfWeights).
	randomWeight := randInt64n(sumOfWeights)
	// Item's accumulated weights are in ascending order, because item's weight >= 0.
	// Binary search rw.items to find first item whose accumulatedWeight > randomWeight
	// The return i is guaranteed to be in range [0, len(rw.items)) because randomWeight < last item's accumulatedWeight
	i := sort.Search(len(rw.items), func(i int) bool { return rw.items[i].accumulatedWeight > randomWeight })
	return rw.items[i].item
}

func (rw *randomWRR) Add(item any, weight int64) {
	accumulatedWeight := weight
	equalWeights := true
	if len(rw.items) > 0 {
		lastItem := rw.items[len(rw.items)-1]
		accumulatedWeight = lastItem.accumulatedWeight + weight
		equalWeights = rw.equalWeights && weight == lastItem.weight
	}
	rw.equalWeights = equalWeights
	rItem := &weightedItem{item: item, weight: weight, accumulatedWeight: accumulatedWeight}
	rw.items = append(rw.items, rItem)
}

func (rw *randomWRR) String() string {
	return fmt.Sprint(rw.items)
}
