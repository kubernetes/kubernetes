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

import "github.com/juju/ratelimit"

type RateLimiter interface {
	// CanAccept returns true if the rate is below the limit, false otherwise
	CanAccept() bool
	// Accept returns once a token becomes available.
	Accept()
	// Stop stops the rate limiter, subsequent calls to CanAccept will return false
	Stop()
}

type tickRateLimiter struct {
	limiter *ratelimit.Bucket
}

// NewTokenBucketRateLimiter creates a rate limiter which implements a token bucket approach.
// The rate limiter allows bursts of up to 'burst' to exceed the QPS, while still maintaining a
// smoothed qps rate of 'qps'.
// The bucket is initially filled with 'burst' tokens, and refills at a rate of 'qps'.
// The maximum number of tokens in the bucket is capped at 'burst'.
func NewTokenBucketRateLimiter(qps float32, burst int) RateLimiter {
	limiter := ratelimit.NewBucketWithRate(float64(qps), int64(burst))
	return &tickRateLimiter{limiter}
}

type fakeRateLimiter struct{}

func NewFakeRateLimiter() RateLimiter {
	return &fakeRateLimiter{}
}

func (t *tickRateLimiter) CanAccept() bool {
	return t.limiter.TakeAvailable(1) == 1
}

// Accept will block until a token becomes available
func (t *tickRateLimiter) Accept() {
	t.limiter.Wait(1)
}

func (t *tickRateLimiter) Stop() {
}

func (t *fakeRateLimiter) CanAccept() bool {
	return true
}

func (t *fakeRateLimiter) Stop() {}

func (t *fakeRateLimiter) Accept() {}
