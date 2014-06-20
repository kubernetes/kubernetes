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
	"math"
	"math/rand"
	"sync"
)

type esSampleItem struct {
	data interface{}
	key  float64
}

type esSampleHeap []esSampleItem

func (self esSampleHeap) Len() int {
	return len(self)
}

func (self esSampleHeap) Less(i, j int) bool {
	return self[i].key < self[j].key
}

func (self esSampleHeap) Swap(i, j int) {
	self[i], self[j] = self[j], self[i]
}

func (self *esSampleHeap) Push(x interface{}) {
	item := x.(esSampleItem)
	*self = append(*self, item)
}

func (self *esSampleHeap) Pop() interface{} {
	old := *self
	item := old[len(old)-1]
	*self = old[:len(old)-1]
	return item
}

type esSampler struct {
	weight  func(interface{}) float64
	samples *esSampleHeap
	maxSize int
	lock    sync.RWMutex
}

func (self *esSampler) Update(d interface{}) {
	self.lock.Lock()
	defer self.lock.Unlock()

	u := rand.Float64()
	key := math.Pow(u, 1.0/self.weight(d))

	if self.samples.Len() < self.maxSize {
		heap.Push(self.samples, esSampleItem{
			data: d,
			key:  key,
		})
		return
	}

	s := *(self.samples)
	min := s[0]

	// The key of the new item is larger than a key in existing item.
	// Add this new item.
	if key > min.key {
		heap.Pop(self.samples)
		heap.Push(self.samples, esSampleItem{
			data: d,
			key:  key,
		})
	}
}

func (self *esSampler) Len() int {
	self.lock.RLock()
	defer self.lock.RUnlock()
	return len(*self.samples)
}

func (self *esSampler) Reset() {
	self.lock.Lock()
	defer self.lock.Unlock()
	self.samples = &esSampleHeap{}
	heap.Init(self.samples)
}

func (self *esSampler) Map(f func(interface{})) {
	self.lock.RLock()
	defer self.lock.RUnlock()

	for _, d := range *self.samples {
		f(d.data)
	}
}

func (self *esSampler) Filter(filter func(d interface{}) bool) {
	self.lock.Lock()
	defer self.lock.Unlock()

	rmlist := make([]int, 0, len(*self.samples))
	for i, d := range *self.samples {
		if filter(d.data) {
			rmlist = append(rmlist, i)
		}
	}

	for _, i := range rmlist {
		heap.Remove(self.samples, i)
	}
}

// ES sampling algorithm described in
//
// Pavlos S. Efraimidis and Paul G. Spirakis. Weighted random sampling with a
// reservoir. Information Processing Letters, 97(5):181 – 185, 2006.
//
// http://dl.acm.org/citation.cfm?id=1138834
func NewESSampler(size int, weight func(interface{}) float64) Sampler {
	s := &esSampleHeap{}
	heap.Init(s)
	return &esSampler{
		maxSize: size,
		samples: s,
		weight:  weight,
	}
}
