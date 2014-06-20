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
	"math/rand"
	"sync"
)

// Reservoir sampling algorithm.
// http://en.wikipedia.org/wiki/Reservoir_sampling
type reservoirSampler struct {
	maxSize      int
	samples      []interface{}
	numInstances int64
	lock         sync.RWMutex
}

func (self *reservoirSampler) Len() int {
	self.lock.RLock()
	defer self.lock.RUnlock()
	return len(self.samples)
}

func (self *reservoirSampler) Reset() {
	self.lock.Lock()
	defer self.lock.Unlock()
	self.samples = make([]interface{}, 0, self.maxSize)
	self.numInstances = 0
}

// Update samples according to http://en.wikipedia.org/wiki/Reservoir_sampling
func (self *reservoirSampler) Update(d interface{}) {
	self.lock.Lock()
	defer self.lock.Unlock()

	self.numInstances++
	if len(self.samples) < self.maxSize {
		self.samples = append(self.samples, d)
		return
	}
	// Randomly generates a number between [0, numInstances).
	// Use this random number, j, as an index.  If j is larger than the
	// reservoir size, we will ignore the current new data.
	// Otherwise replace the jth element in reservoir with the new data.
	j := rand.Int63n(self.numInstances)
	if j < int64(len(self.samples)) {
		self.samples[int(j)] = d
	}
}

func (self *reservoirSampler) Map(f func(d interface{})) {
	self.lock.RLock()
	defer self.lock.RUnlock()

	for _, d := range self.samples {
		f(d)
	}
}

// Once an element is removed, the probability of sampling an observation will
// be increased.  Removing all elements in the sampler has the same effect as
// calling Reset(). However, it will not guarantee the uniform probability of
// all unfiltered samples.
func (self *reservoirSampler) Filter(filter func(d interface{}) bool) {
	self.lock.Lock()
	defer self.lock.Unlock()
	rmlist := make([]int, 0, len(self.samples))
	for i, d := range self.samples {
		if filter(d) {
			rmlist = append(rmlist, i)
		}
	}

	for _, i := range rmlist {
		// slice trick: remove the ith element without preserving the order
		self.samples[i] = self.samples[len(self.samples)-1]
		self.samples = self.samples[:len(self.samples)-1]
	}
	self.numInstances -= int64(len(rmlist))
}

func NewReservoirSampler(reservoirSize int) Sampler {
	return &reservoirSampler{
		maxSize: reservoirSize,
	}
}
