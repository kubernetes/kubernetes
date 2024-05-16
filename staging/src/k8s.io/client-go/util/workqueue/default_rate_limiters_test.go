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
	"testing"
	"time"
)

func TestItemExponentialFailureRateLimiter(t *testing.T) {
	limiter := NewItemExponentialFailureRateLimiter(1*time.Millisecond, 1*time.Second)

	if e, a := 1*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 2*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 4*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 8*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 16*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 5, limiter.NumRequeues("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	if e, a := 1*time.Millisecond, limiter.When("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 2*time.Millisecond, limiter.When("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 2, limiter.NumRequeues("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	limiter.Forget("one")
	if e, a := 0, limiter.NumRequeues("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 1*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

}

func TestItemExponentialFailureRateLimiterOverFlow(t *testing.T) {
	limiter := NewItemExponentialFailureRateLimiter(1*time.Millisecond, 1000*time.Second)
	for i := 0; i < 5; i++ {
		limiter.When("one")
	}
	if e, a := 32*time.Millisecond, limiter.When("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	for i := 0; i < 1000; i++ {
		limiter.When("overflow1")
	}
	if e, a := 1000*time.Second, limiter.When("overflow1"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	limiter = NewItemExponentialFailureRateLimiter(1*time.Minute, 1000*time.Hour)
	for i := 0; i < 2; i++ {
		limiter.When("two")
	}
	if e, a := 4*time.Minute, limiter.When("two"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	for i := 0; i < 1000; i++ {
		limiter.When("overflow2")
	}
	if e, a := 1000*time.Hour, limiter.When("overflow2"); e != a {
		t.Errorf("expected %v, got %v", e, a)
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
	limiter := NewMaxOfRateLimiter(
		NewItemFastSlowRateLimiter(5*time.Millisecond, 3*time.Second, 3),
		NewItemExponentialFailureRateLimiter(1*time.Millisecond, 1*time.Second),
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
