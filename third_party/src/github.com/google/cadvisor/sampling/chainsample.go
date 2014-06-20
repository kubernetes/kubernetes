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
	"log"
	"math/rand"
	"sync"

	"github.com/kr/pretty"
)

type empty struct{}

// Randomly generate number [start,end) except @except.
func randInt64Except(start, end int64, except map[int64]empty) int64 {
	n := end - start
	ret := rand.Int63n(n) + start
	for _, ok := except[ret]; ok; _, ok = except[ret] {
		ret = rand.Int63n(n) + start
	}
	return ret
}

// Basic idea:
// Every observation will have a sequence number as its id.
// Suppose we want to sample k observations within latest n observations
// At first, we generated k random numbers in [0,n). These random numbers
// will be used as ids of observations that will be sampled.
type chainSampler struct {
	sampleSize int
	windowSize int64

	// Every observation will have a sequence number starting from 1.
	// The sequence number must increase by one for each observation.
	numObservations int64

	// All samples stored as id -> value.
	samples map[int64]interface{}

	// The set of id of future observations.
	futureSamples map[int64]empty

	// The chain of samples: old observation id -> future observation id.
	// When the old observation expires, the future observation will be
	// stored as a sample.
	sampleChain map[int64]int64

	// Replacements are: observations whose previous sample is not expired
	// id->value.
	replacements map[int64]interface{}
	lock         sync.RWMutex
}

func (self *chainSampler) initFutureSamples() {
	for i := 0; i < self.sampleSize; i++ {
		n := randInt64Except(1, self.windowSize+1, self.futureSamples)
		self.futureSamples[n] = empty{}
	}
}

func (self *chainSampler) arrive(seqNum int64, obv interface{}) {
	if _, ok := self.futureSamples[seqNum]; !ok {
		// If this observation is not selected, ignore it.
		return
	}

	delete(self.futureSamples, seqNum)

	if len(self.samples) < self.sampleSize {
		self.samples[seqNum] = obv
	}
	self.replacements[seqNum] = obv

	// Select a future observation which will replace current observation
	// when it expires.
	futureSeqNum := randInt64Except(seqNum+1, seqNum+self.windowSize+1, self.futureSamples)
	self.futureSamples[futureSeqNum] = empty{}
	self.sampleChain[seqNum] = futureSeqNum
}

func (self *chainSampler) expireAndReplace() {
	expSeqNum := self.numObservations - self.windowSize
	if _, ok := self.samples[expSeqNum]; !ok {
		// No sample expires
		return
	}
	delete(self.samples, expSeqNum)
	// There must be a replacement, otherwise panic.
	replacementSeqNum := self.sampleChain[expSeqNum]
	// The sequence number must increase by one for each observation.
	replacement, ok := self.replacements[replacementSeqNum]
	if !ok {
		log.Printf("cannot find %v. which is the replacement of %v\n", replacementSeqNum, expSeqNum)
		pretty.Printf("chain: %# v\n", self)
		panic("Should never occur!")
	}
	// This observation must have arrived before.
	self.samples[replacementSeqNum] = replacement
}

func (self *chainSampler) Update(obv interface{}) {
	self.lock.Lock()
	defer self.lock.Unlock()

	self.numObservations++
	self.arrive(self.numObservations, obv)
	self.expireAndReplace()
}

func (self *chainSampler) Len() int {
	self.lock.RLock()
	defer self.lock.RUnlock()
	return len(self.samples)
}

func (self *chainSampler) Reset() {
	self.lock.Lock()
	defer self.lock.Unlock()
	self.numObservations = 0
	self.samples = make(map[int64]interface{}, self.sampleSize)
	self.futureSamples = make(map[int64]empty, self.sampleSize*2)
	self.sampleChain = make(map[int64]int64, self.sampleSize*2)
	self.replacements = make(map[int64]interface{}, self.sampleSize*2)
	self.initFutureSamples()
}

func (self *chainSampler) Map(f func(d interface{})) {
	self.lock.RLock()
	defer self.lock.RUnlock()

	for seqNum, obv := range self.samples {
		if _, ok := obv.(int); !ok {
			pretty.Printf("Seq %v. WAT: %# v\n", seqNum, obv)
		}
		f(obv)
	}
}

// NOT SUPPORTED
func (self *chainSampler) Filter(filter func(d interface{}) bool) {
	return
}

// Chain sampler described in
// Brian Babcok, Mayur Datar and Rajeev Motwani,
// Sampling From a Moving Window Over Streaming Data
func NewChainSampler(sampleSize, windowSize int) Sampler {
	sampler := &chainSampler{
		sampleSize:    sampleSize,
		windowSize:    int64(windowSize),
		samples:       make(map[int64]interface{}, sampleSize),
		futureSamples: make(map[int64]empty, sampleSize*2),
		sampleChain:   make(map[int64]int64, sampleSize*2),
		replacements:  make(map[int64]interface{}, sampleSize*2),
	}
	sampler.initFutureSamples()
	return sampler
}
