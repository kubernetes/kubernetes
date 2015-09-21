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
	"testing"
	"time"
)

func TestBasicThrottle(t *testing.T) {
	r := NewTokenBucketRateLimiter(1, 3)
	for i := 0; i < 3; i++ {
		if !r.CanAccept() {
			t.Error("unexpected false accept")
		}
	}
	if r.CanAccept() {
		t.Error("unexpected true accept")
	}
}

func TestIncrementThrottle(t *testing.T) {
	r := NewTokenBucketRateLimiter(1, 1)
	if !r.CanAccept() {
		t.Error("unexpected false accept")
	}
	if r.CanAccept() {
		t.Error("unexpected true accept")
	}

	// Allow to refill
	time.Sleep(2 * time.Second)

	if !r.CanAccept() {
		t.Error("unexpected false accept")
	}
}

func TestThrottle(t *testing.T) {
	r := NewTokenBucketRateLimiter(10, 5)

	// Should consume 5 tokens immediately, then
	// the remaining 11 should take at least 1 second (0.1s each)
	expectedFinish := time.Now().Add(time.Second * 1)
	for i := 0; i < 16; i++ {
		r.Accept()
	}
	if time.Now().Before(expectedFinish) {
		t.Error("rate limit was not respected, finished too early")
	}
}
