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
	"sync"
	"time"

	"github.com/golang/glog"
)

// PeriodicRunner runs a function on a regular period.
// Callers can manually trigger runs, which will reset the period, but those
// manual runs are rate-limited.
// TODO: inject a clock interface for testing.
type PeriodicRunner struct {
	minInterval time.Duration // the min time between runs, modulo bursts
	maxInterval time.Duration // the max time between runs

	mu      sync.Mutex
	fn      func()
	lastRun time.Time
	timer   *time.Timer
	limiter RateLimiter
}

// NewPeriodicRunner creates a new PeriodicRunner.
func NewPeriodicRunner(fn func(), minInterval, maxInterval time.Duration, burst int) *PeriodicRunner {
	pr := &PeriodicRunner{
		fn:          fn,
		minInterval: minInterval,
		maxInterval: maxInterval,
	}
	if minInterval == 0 {
		pr.limiter = &fakeAlwaysRateLimiter{}
	} else {
		rps := float32(time.Second) / float32(minInterval)
		pr.limiter = NewTokenBucketRateLimiter(rps, burst)
	}
	return pr
}

// Run starts the periodic timer. This is expected to be called as a go-routine.
func (pr *PeriodicRunner) Run(stopCh <-chan struct{}) {
	pr.timer = time.NewTimer(pr.maxInterval)
	for {
		select {
		case <-pr.timer.C:
			pr.tick()
		case <-stopCh:
			pr.stop()
			return
		}
	}
}

func (pr *PeriodicRunner) stop() {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	pr.limiter.Stop()
	pr.timer.Stop()
}

func (pr *PeriodicRunner) tick() {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	pr.callFunction()
	pr.timer.Reset(pr.maxInterval)
}

// CallFunction calls the function as soon as possible.
// If this is called while Run() is not running, the call may be deferred
// indefinitely.
// This function is non-blocking.
func (pr *PeriodicRunner) CallFunction() {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	pr.callFunction()
}

// assumes the lock is held.
func (pr *PeriodicRunner) callFunction() {
	if pr.limiter.TryAccept() {
		pr.fn()
		pr.lastRun = time.Now()
		return
	}

	// It can't run right now, figure out when it can run next.
	elapsed := time.Since(pr.lastRun)
	asap := pr.minInterval - elapsed
	next := pr.maxInterval - elapsed
	if next <= asap {
		// just let the periodi timer catch it.
		glog.V(3).Infof("running too often: eta %v", next)
		return
	}

	// Set tht timer for ASAP, but don't drain here.
	// Assuming Run() is running, it might get a delivery in the mean time,
	// but that is OK.
	pr.timer.Stop()
	pr.timer.Reset(asap)
	glog.V(3).Infof("running too often: eta: %v", asap)
}
