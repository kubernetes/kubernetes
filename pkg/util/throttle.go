/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package util

import (
	"sync"
	"time"
)

type RateLimiter interface {
	// CanAccept returns true if the rate is below the limit, false otherwise
	CanAccept() bool
	// Accept returns once a token becomes available.
	Accept()
	// Stop stops the rate limiter, subsequent calls to CanAccept will return false
	Stop()
}

type tickRateLimiter struct {
	lock   sync.Mutex
	tokens chan bool
	ticker <-chan time.Time
	stop   chan bool
}

// NewTokenBucketRateLimiter creates a rate limiter which implements a token bucket approach.
// The rate limiter allows bursts of up to 'burst' to exceed the QPS, while still maintaining a
// smoothed qps rate of 'qps'.
// The bucket is initially filled with 'burst' tokens, the rate limiter spawns a go routine
// which refills the bucket with one token at a rate of 'qps'.  The maximum number of tokens in
// the bucket is capped at 'burst'.
// When done with the limiter, Stop() must be called to halt the associated goroutine.
func NewTokenBucketRateLimiter(qps float32, burst int) RateLimiter {
	ticker := time.Tick(time.Duration(float32(time.Second) / qps))
	rate := newTokenBucketRateLimiterFromTicker(ticker, burst)
	go rate.run()
	return rate
}

type fakeRateLimiter struct{}

func NewFakeRateLimiter() RateLimiter {
	return &fakeRateLimiter{}
}

func newTokenBucketRateLimiterFromTicker(ticker <-chan time.Time, burst int) *tickRateLimiter {
	if burst < 1 {
		panic("burst must be a positive integer")
	}
	rate := &tickRateLimiter{
		tokens: make(chan bool, burst),
		ticker: ticker,
		stop:   make(chan bool),
	}
	for i := 0; i < burst; i++ {
		rate.tokens <- true
	}
	return rate
}

func (t *tickRateLimiter) CanAccept() bool {
	select {
	case <-t.tokens:
		return true
	default:
		return false
	}
}

// Accept will block until a token becomes available
func (t *tickRateLimiter) Accept() {
	<-t.tokens
}

func (t *tickRateLimiter) Stop() {
	close(t.stop)
}

func (r *tickRateLimiter) run() {
	for {
		if !r.step() {
			break
		}
	}
}

func (r *tickRateLimiter) step() bool {
	select {
	case <-r.ticker:
		r.increment()
		return true
	case <-r.stop:
		return false
	}
}

func (t *tickRateLimiter) increment() {
	// non-blocking send
	select {
	case t.tokens <- true:
	default:
	}
}

func (t *fakeRateLimiter) CanAccept() bool {
	return true
}

func (t *fakeRateLimiter) Stop() {}

func (t *fakeRateLimiter) Accept() {}
