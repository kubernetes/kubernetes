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

package iptables

import (
	"sync"
	"time"

	"k8s.io/client-go/util/flowcontrol"

	"github.com/golang/glog"
)

// PeriodicRunner runs a function on a regular period.  Callers can manually
// trigger runs, which will reset the period.  Manual runs are rate-limited.
// TODO: pass in a name string?
// TODO: inject a clock interface for testing
// TODO: inject a logger interface to decouple from glog
type PeriodicRunner struct {
	minInterval time.Duration // the min time between runs, modulo bursts
	maxInterval time.Duration // the max time between runs

	mu      sync.Mutex  // guards runs of fn and all mutations
	fn      func()      // function to run
	lastRun time.Time   // time since last run
	timer   *time.Timer // timer for periodic runs
	limiter rateLimiter // rate limiter for on-demand runs
}

// designed so that flowcontrol.RateLimiter satisfies
type rateLimiter interface {
	TryAccept() bool
	Stop()
}

type nullLimiter struct{}

func (nullLimiter) TryAccept() bool {
	return true
}

func (nullLimiter) Stop() {}

// NewPeriodicRunner creates a new PeriodicRunner.
func NewPeriodicRunner(fn func(), minInterval time.Duration, maxInterval time.Duration, burst int) *PeriodicRunner {
	pr := &PeriodicRunner{
		fn:          fn,
		minInterval: minInterval,
		maxInterval: maxInterval,
	}
	if minInterval == 0 {
		pr.limiter = nullLimiter{}
	} else {
		// minInterval is a duration, typically in seconds but could be fractional
		rps := float32(time.Second) / float32(minInterval)
		// allow burst updates in short succession
		pr.limiter = flowcontrol.NewTokenBucketRateLimiter(rps, burst)
	}
	return pr
}

// Run the periodic timer.  This is expected to be called as a goroutine.
func (pr *PeriodicRunner) Loop(stop <-chan struct{}) {
	pr.timer = time.NewTimer(pr.maxInterval)
	for {
		select {
		case <-stop:
			pr.stop()
			return
		case <-pr.timer.C:
			pr.tick()
		}
	}
}

// assumes the lock is held
func (pr *PeriodicRunner) stop() {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	pr.limiter.Stop()
	pr.timer.Stop()
}

// assumes the lock is held
func (pr *PeriodicRunner) tick() {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	pr.run()
	pr.timer.Reset(pr.maxInterval)
}

// Run the function as soon as possible.  If this is called while Loop is not
// running, the call may be deferred indefinitely.
func (pr *PeriodicRunner) Run() {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	pr.run()
}

// assumes the lock is held
func (pr *PeriodicRunner) run() {
	if pr.limiter.TryAccept() {
		pr.fn()
		pr.lastRun = time.Now()
		return
	}

	// It can't run right now, figure out when it can run next.

	elapsed := time.Since(pr.lastRun) // how long since last run
	asap := pr.minInterval - elapsed  // time to next possible run
	next := pr.maxInterval - elapsed  // time to next periodic run

	if next <= asap {
		// just let the periodic timer catch it
		glog.V(3).Infof("running too often: eta %v", next)
		return
	}

	// Set the timer for ASAP, but don't drain here.  Assuming Loop is running,
	// it might get a delivery in the mean time, but that is OK.
	pr.timer.Stop()
	pr.timer.Reset(asap)
	glog.V(3).Infof("running too often: eta %v", asap)
}
