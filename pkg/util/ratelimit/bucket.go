/*
Copyright 2016 The Kubernetes Authors.

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

package ratelimit

import (
	"sync"
	"time"
)

// Bucket models a token bucket
type Bucket struct {
	unitsPerNano float64
	nanosPerUnit float64
	capacity     int64

	mutex      sync.Mutex
	available  int64
	lastRefill int64
}

// NewBucketWithRate creates a new token bucket, with maximum and initial capacity, and a refill rate of qps
func NewBucketWithRate(qps float64, capacity int64) *Bucket {
	unitsPerNano := qps / 1E9
	nanosPerUnit := 1E9 / qps
	b := &Bucket{
		unitsPerNano: unitsPerNano,
		nanosPerUnit: nanosPerUnit,
		capacity:     capacity,
		available:    capacity,
		lastRefill:   time.Now().UnixNano(),
	}
	return b
}

// Take takes n units from the bucket, reducing the available quantity even below zero,
// but then returns the amount of time we should wait
func (b *Bucket) Take(n int64) time.Duration {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	var d time.Duration
	if b.available >= n {
		// Fast path when bucket has sufficient availability before refilling
	} else {
		b.refill()

		if b.available < n {
			deficit := n - b.available
			d = time.Duration(int64(float64(deficit) * b.nanosPerUnit))
		}
	}

	b.available -= n

	return d
}

// TakeAvailable immediately takes whatever quantity is available, up to max
func (b *Bucket) TakeAvailable(max int64) int64 {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	var took int64
	if b.available >= max {
		// Fast path when bucket has sufficient availability before refilling
		took = max
	} else {
		b.refill()

		took = b.available

		if took < 0 {
			took = 0
		} else if took > max {
			took = max
		}
	}

	if took > 0 {
		b.available -= took
	}

	return took
}

// Wait combines a call to Take with a sleep call
func (b *Bucket) Wait(n int64) {
	d := b.Take(n)
	if d != 0 {
		time.Sleep(d)
	}
}

// Capacity returns the maximum capacity of the bucket
func (b *Bucket) Capacity() int64 {
	return b.capacity
}

// Available returns the quantity available in the bucket (which may be negative)
func (b *Bucket) Available() int64 {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	b.refill()

	return b.available
}

// refill replenishes the bucket based on elapsed time; mutex must be held
func (b *Bucket) refill() {
	// Note that we really want a monotonic clock here, but go says no:
	// https://github.com/golang/go/issues/12914
	now := time.Now().UnixNano()

	b.refillAtTimestamp(now)
}

// refillAtTimestamp is the logic of the refill function, for testing
func (b *Bucket) refillAtTimestamp(now int64) {
	nanosSinceLastRefill := now - b.lastRefill
	if nanosSinceLastRefill < 0 {
		// we really want monotonic
		return
	}

	// Compute whole number of units that has flowed into bucket
	refill := int64(float64(nanosSinceLastRefill) * b.unitsPerNano)
	if refill == 0 {
		return
	}

	// Refill with overflow
	b.available += refill
	if b.available > b.capacity {
		b.available = b.capacity
	}

	// Update time based on units that flowed into bucket
	b.lastRefill += int64(float64(refill) * b.nanosPerUnit)
}
