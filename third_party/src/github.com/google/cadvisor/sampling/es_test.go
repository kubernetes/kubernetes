// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package sampling

import (
	"container/heap"
	"math/rand"
	"testing"

	"github.com/kr/pretty"
)

// This should be a min heap
func TestESSampleHeap(t *testing.T) {
	h := &esSampleHeap{}
	heap.Init(h)
	min := 5.0
	N := 10

	for i := 0; i < N; i++ {
		key := rand.Float64()
		if key < min {
			min = key
		}
		heap.Push(h, esSampleItem{nil, key})
	}
	l := *h
	if l[0].key != min {
		t.Errorf("not a min heap")
		pretty.Printf("min=%v\nheap=%# v\n", min, l)
	}
}

func TestESSampler(t *testing.T) {
	reservoirSize := 10
	numObvs := 10 * reservoirSize
	numSampleRounds := 100 * numObvs

	weight := func(d interface{}) float64 {
		n := d.(int)
		return float64(n + 1)
	}
	s := NewESSampler(reservoirSize, weight)
	hist := make(map[int]int, numObvs)
	for i := 0; i < numSampleRounds; i++ {
		sampleStream(hist, numObvs, s)
	}

	diff := 2
	wrongOrderedItems := make([]int, 0, numObvs)
	threshold := 1.05
	for i := 0; i < numObvs-diff; i++ {
		// Item with smaller weight should have lower probability to be selected.
		n1 := hist[i]
		n2 := hist[i+diff]
		if n1 > n2 {
			if float64(n1) > float64(n2)*threshold {
				wrongOrderedItems = append(wrongOrderedItems, i)
			}
		}
	}
	if float64(len(wrongOrderedItems)) > float64(numObvs)*0.05 {
		for _, i := range wrongOrderedItems {
			n1 := hist[i]
			n2 := hist[i+diff]
			t.Errorf("item with weight %v is selected %v times; while item with weight %v is selected %v times", i, n1, i+diff, n2)
		}
	}
}
