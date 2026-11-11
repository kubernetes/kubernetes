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

package workqueue

import (
	"math"
	"strconv"
	"testing"
	"time"
)

var (
	randMin = func(int64) int64 { return 0 }
	randMax = func(n int64) int64 { return n - 1 }
)

func TestItemExponentialFailureRateLimiter(t *testing.T) {
	limiter := NewItemExponentialFailureRateLimiter(1*time.Millisecond, 10*time.Millisecond).(*ItemExponentialFailureRateLimiter)
	limiter.int64N = randMax

	// The largest possible delay follows baseDelay*2^n, capped at maxDelay.
	for i, e := range []time.Duration{1, 2, 4, 8, 10, 10} {
		if e, a := e*time.Millisecond-1, limiter.When("one"); e != a {
			t.Errorf("attempt %d: expected %v, got %v", i, e, a)
		}
	}
	if e, a := 6, limiter.NumRequeues("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// The smallest possible delay is zero.
	limiter.int64N = randMin
	if e, a := time.Duration(0), limiter.When("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 1, limiter.NumRequeues("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	limiter.Forget("one")
	if e, a := 0, limiter.NumRequeues("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	limiter.int64N = randMax
	if e, a := 1*time.Millisecond-1, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestItemExponentialFailureRateLimiterOverFlow(t *testing.T) {
	limiter := NewItemExponentialFailureRateLimiter(1*time.Millisecond, 1000*time.Second).(*ItemExponentialFailureRateLimiter)
	limiter.int64N = randMax
	for i := 0; i < 1000; i++ {
		limiter.When("overflow1")
	}
	if e, a := 1000*time.Second-1, limiter.When("overflow1"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	limiter = NewItemExponentialFailureRateLimiter(1*time.Minute, math.MaxInt64).(*ItemExponentialFailureRateLimiter)
	limiter.int64N = randMax
	for i := range 100 {
		if a := limiter.When("overflow2"); a < 0 {
			t.Fatalf("attempt %d: negative delay %v, overflowed", i, a)
		}
	}
	if e, a := time.Duration(math.MaxInt64-1), limiter.When("overflow2"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestItemExponentialFailureRateLimiterNonPositiveDelays(t *testing.T) {
	for name, limiter := range map[string]RateLimiter{
		"zero base": NewItemExponentialFailureRateLimiter(0, time.Second),
		"zero max":  NewItemExponentialFailureRateLimiter(time.Millisecond, 0),
	} {
		for i := range 5 {
			if e, a := time.Duration(0), limiter.When("one"); e != a {
				t.Errorf("%s attempt %d: expected %v, got %v", name, i, e, a)
			}
		}
	}
}

// TestItemExponentialFailureRateLimiterSpreadsRetries checks that items failing
// at the same time get different delays within the backoff window, which is
// the point of jitter.
func TestItemExponentialFailureRateLimiterSpreadsRetries(t *testing.T) {
	const items = 1000
	base, maxDelay := 5*time.Millisecond, 1000*time.Second
	limiter := NewTypedItemExponentialFailureRateLimiter[string](base, maxDelay)
	for attempt := range 10 {
		ceiling := min(maxDelay, base<<attempt)
		distinct := map[time.Duration]struct{}{}
		for i := range items {
			d := limiter.When(strconv.Itoa(i))
			if d < 0 || d >= ceiling {
				t.Fatalf("attempt %d: delay %v outside [0, %v)", attempt, d, ceiling)
			}
			distinct[d] = struct{}{}
		}
		// Collisions are possible but should be rare at nanosecond resolution.
		if len(distinct) < items*9/10 {
			t.Errorf("attempt %d: only %d distinct delays for %d items", attempt, len(distinct), items)
		}
	}
}

func TestItemFastSlowRateLimiter(t *testing.T) {
	limiter := NewItemFastSlowRateLimiter(5*time.Millisecond, 10*time.Second, 3)

	if e, a := 5*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 5*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 5*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 10*time.Second, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 10*time.Second, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 5, limiter.NumRequeues("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	if e, a := 5*time.Millisecond, limiter.When("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 5*time.Millisecond, limiter.When("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 2, limiter.NumRequeues("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	limiter.Forget("one")
	if e, a := 0, limiter.NumRequeues("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 5*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

}

func TestMaxOfRateLimiter(t *testing.T) {
	exponential := NewItemExponentialFailureRateLimiter(1*time.Millisecond, 1*time.Second).(*ItemExponentialFailureRateLimiter)
	exponential.int64N = randMin
	limiter := NewMaxOfRateLimiter(
		NewItemFastSlowRateLimiter(5*time.Millisecond, 3*time.Second, 3),
		exponential,
	)

	if e, a := 5*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 5*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 5*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 3*time.Second, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 3*time.Second, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 5, limiter.NumRequeues("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	if e, a := 5*time.Millisecond, limiter.When("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 5*time.Millisecond, limiter.When("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 2, limiter.NumRequeues("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	limiter.Forget("one")
	if e, a := 0, limiter.NumRequeues("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 5*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

}

func TestWithMaxWaitRateLimiter(t *testing.T) {
	limiter := NewWithMaxWaitRateLimiter(NewStepRateLimiter(5*time.Millisecond, 1000*time.Second, 100), 500*time.Second)
	for i := 0; i < 100; i++ {
		if e, a := 5*time.Millisecond, limiter.When(i); e != a {
			t.Errorf("expected %v, got %v ", e, a)
		}
	}

	for i := 100; i < 200; i++ {
		if e, a := 500*time.Second, limiter.When(i); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
	}
}

var _ RateLimiter = &StepRateLimiter{}

func NewStepRateLimiter(baseDelay time.Duration, maxDelay time.Duration, threshold int) RateLimiter {
	return &StepRateLimiter{
		baseDelay: baseDelay,
		maxDelay:  maxDelay,
		threshold: threshold,
	}
}

type StepRateLimiter struct {
	count     int
	threshold int
	baseDelay time.Duration
	maxDelay  time.Duration
}

func (r *StepRateLimiter) When(item interface{}) time.Duration {
	r.count += 1
	if r.count <= r.threshold {
		return r.baseDelay
	}
	return r.maxDelay
}

func (r *StepRateLimiter) NumRequeues(item interface{}) int {
	return 0
}

func (r *StepRateLimiter) Forget(item interface{}) {
}
