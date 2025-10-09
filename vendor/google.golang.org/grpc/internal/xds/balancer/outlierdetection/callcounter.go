/*
 *
 * Copyright 2022 gRPC authors.
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

package outlierdetection

import (
	"sync/atomic"
)

type bucket struct {
	numSuccesses uint32
	numFailures  uint32
}

func newCallCounter() *callCounter {
	cc := &callCounter{
		inactiveBucket: &bucket{},
	}
	cc.activeBucket.Store(&bucket{})
	return cc
}

// callCounter has two buckets, which each count successful and failing RPC's.
// The activeBucket is used to actively count any finished RPC's, and the
// inactiveBucket is populated with this activeBucket's data every interval for
// use by the Outlier Detection algorithm.
type callCounter struct {
	// activeBucket updates every time a call finishes (from picker passed to
	// Client Conn), so protect pointer read with atomic load of the pointer
	// so picker does not have to grab a mutex per RPC, the critical path.
	activeBucket   atomic.Pointer[bucket]
	inactiveBucket *bucket
}

func (cc *callCounter) clear() {
	cc.activeBucket.Store(&bucket{})
	cc.inactiveBucket = &bucket{}
}

// "When the timer triggers, the inactive bucket is zeroed and swapped with the
// active bucket. Then the inactive bucket contains the number of successes and
// failures since the last time the timer triggered. Those numbers are used to
// evaluate the ejection criteria." - A50.
func (cc *callCounter) swap() {
	ib := cc.inactiveBucket
	*ib = bucket{}
	ab := cc.activeBucket.Swap(ib)
	cc.inactiveBucket = &bucket{
		numSuccesses: atomic.LoadUint32(&ab.numSuccesses),
		numFailures:  atomic.LoadUint32(&ab.numFailures),
	}
}
