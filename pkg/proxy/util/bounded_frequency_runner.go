/*
Copyright 2025 The Kubernetes Authors.

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
	"time"

	"golang.org/x/time/rate"
	"k8s.io/klog/v2"
)

type BoundedFrequencyRunner struct {
	maxInterval time.Duration
	rateLimiter *rate.Limiter
	fn          func()
	queue       chan struct{}
}

// NewBoundedFrequencyRunner creates a new BoundedFrequencyRunner instance,
// which will manage runs of the specified function.
//
// All runs will be async to the caller of BoundedFrequencyRunner.Run, but
// multiple runs are serialized. If the function needs to hold locks, it must
// take them internally.
//
// Runs of the function will have at least minInterval between them (from
// completion to next start), except that up to bursts may be allowed.  Burst
// runs are "accumulated" over time, one per minInterval up to burstRuns total.
// Burst runs only occur if the previous execution of `fn` has completed.
// This can be used, for example, to mitigate the impact of expensive operations
// being called in response to user-initiated operations. Run requests that
// would violate the minInterval are coalesced and run at the next opportunity.
// The function will be run at least once per maxInterval. For example, this can
// force periodic refreshes of state in the absence of anyone calling Run.
//
// Examples:
//
// NewBoundedFrequencyRunner("name", fn, time.Second, 5*time.Second)
// - fn will have at least 1 second between runs
// - fn will have no more than 5 seconds between runs
//
// NewBoundedFrequencyRunner("name", fn, 3*time.Second, 10*time.Second, 3)
// - fn will have at least 3 seconds between runs, with up to 3 burst runs
// - fn will have no more than 10 seconds between runs
//
// The maxInterval must be greater than or equal to the minInterval,  If the
// caller passes a maxInterval less than minInterval, this function will panic.
func NewBoundedFrequencyRunner(name string, fn func(), minInterval, maxInterval time.Duration, burst int) *BoundedFrequencyRunner {
	if maxInterval < minInterval {
		panic("maxInterval must be greater or equal than minInterval")
	}

	return &BoundedFrequencyRunner{
		maxInterval: maxInterval,
		rateLimiter: rate.NewLimiter(rate.Every(minInterval), burst),
		queue:       make(chan struct{}, 1),
		fn:          fn,
	}

}

func (bfr *BoundedFrequencyRunner) Loop(stop <-chan struct{}) {
	defer close(bfr.queue)
	klog.V(3).Infof("Loop running")

	// execute fn() at least every maxInterval
	maxIntervalTicker := time.NewTicker(bfr.maxInterval)
	defer maxIntervalTicker.Stop()

	for {
		select {
		case <-stop:
			return
		case <-maxIntervalTicker.C:
		case <-bfr.queue:
		}
		bfr.fn()
		// reset the intervals
		maxIntervalTicker.Reset(bfr.maxInterval)
	}

}

func (bfr *BoundedFrequencyRunner) RetryAfter(interval time.Duration) {
	go func() {
		time.Sleep(interval)
		bfr.Run()
	}()
}

func (bfr *BoundedFrequencyRunner) Run() {
	if bfr.rateLimiter.Allow() {
		// add a token to the channel if it is empty
		select {
		case bfr.queue <- struct{}{}:
		default:
		}
	}
}
