/*
Copyright 2017 The Kubernetes Authors.

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

package flowcontrol

import (
	"context"
	"errors"
	"math"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
)

type Limiter interface {
	TakeAvailable(int64) int64
	Capacity() int64
	Available() int64
	Wait(int64)
	WaitContext(context.Context, int64) error
}

func NewBucketLimiter(rate float64, capacity int64) (Limiter, error) {
	return NewBucketLimiterWithClock(rate, capacity, clock.RealClock{})
}

func NewBucketLimiterWithClock(rate float64, capacity int64, clock clock.Clock) (Limiter, error) {
	fill, quantum, err := calculateQuantum(rate, capacity)
	if err != nil {
		return nil, err
	}
	return &bucketLimiter{
		rate:      rate,
		capacity:  capacity,
		available: capacity,
		fill:      fill,
		quantum:   quantum,
		start:     clock.Now(),
		clock:     clock,
		mtx:       new(sync.Mutex),
	}, nil
}

type bucketLimiter struct {
	rate      float64
	capacity  int64
	fill      time.Duration
	quantum   int64
	available int64
	ticks     int64
	start     time.Time
	clock     clock.Clock
	mtx       *sync.Mutex
}

func (bl *bucketLimiter) TakeAvailable(count int64) int64 {
	if count <= 0 {
		return 0
	}

	bl.mtx.Lock()
	defer bl.mtx.Unlock()

	bl.adjust(bl.clock.Now())

	if bl.available <= 0 {
		return 0
	}

	if count > bl.available {
		count = bl.available
	}

	bl.available -= count
	return count
}

func (bl *bucketLimiter) Capacity() int64 {
	bl.mtx.Lock()
	defer bl.mtx.Unlock()
	return bl.capacity
}

func (bl *bucketLimiter) Available() int64 {
	bl.mtx.Lock()
	defer bl.mtx.Unlock()
	bl.adjust(bl.clock.Now())
	return bl.available
}

func (bl *bucketLimiter) Wait(count int64) {
	bl.WaitContext(context.Background(), count)
}

func (bl *bucketLimiter) WaitContext(ctx context.Context, count int64) error {
	if count <= 0 {
		return nil
	}
	if d := bl.take(count); d > 0 {
		return SleepContext(ctx, bl.clock, d)
	}
	return nil
}

func (bl *bucketLimiter) take(count int64) time.Duration {
	bl.mtx.Lock()
	defer bl.mtx.Unlock()

	now := bl.clock.Now()
	current := bl.adjust(now)

	bl.available -= count

	if bl.available >= 0 {
		return 0
	}

	tick := current + (bl.quantum-bl.available-1)/bl.quantum
	stop := bl.start.Add(time.Duration(tick) * bl.fill)

	return stop.Sub(now)
}

func (bl *bucketLimiter) adjust(now time.Time) int64 {
	current := int64(now.Sub(bl.start) / bl.fill)

	if bl.available >= bl.capacity {
		return current
	}

	bl.available += (current - bl.ticks) * bl.quantum
	if bl.available > bl.capacity {
		bl.available = bl.capacity
	}
	bl.ticks = current

	return current
}

func calculateQuantum(rate float64, capacity int64) (time.Duration, int64, error) {
	for quantum := int64(1); quantum < 1<<50; quantum = nextQuantum(quantum) {
		fill := time.Duration(1e9 * float64(quantum) / rate)
		if fill <= 0 {
			continue
		}

		if delta := math.Abs(1e9*float64(quantum)/float64(fill) - rate); delta/rate <= 0.01 {
			return fill, quantum, nil
		}
	}
	return 0, 0, errors.New("bucket limiter: unable to find appropriate quantum")
}

func nextQuantum(quantum int64) int64 {
	next := quantum * 11 / 10
	if next == quantum {
		return next + 1
	}
	return next
}
