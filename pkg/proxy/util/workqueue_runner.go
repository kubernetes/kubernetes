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
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

// WorkQueueRunner is a replacement for BoundedFrequencyRunner that uses workqueue
// to manage runs of a user-provided function with rate limiting capabilities.
type WorkQueueRunner struct {
	name        string
	fn          func()
	minInterval time.Duration
	maxInterval time.Duration
	queue       workqueue.DelayingInterface
	queueKey    string
	stopCh      chan struct{}
	lock        sync.Mutex
	lastRun     time.Time
}

// NewWorkQueueRunner creates a new WorkQueueRunner instance, which will manage
// runs of the specified function using workqueue.
//
// All runs will be async to the caller of WorkQueueRunner.Run, but
// multiple runs are serialized. If the function needs to hold locks, it must
// take them internally.
//
// Runs of the function will have at least minInterval between them (from
// completion to next start). Run requests that would violate the minInterval
// are coalesced and run at the next opportunity.
//
// The function will be run at least once per maxInterval. For example, this can
// force periodic refreshes of state in the absence of anyone calling Run.
//
// The maxInterval must be greater than or equal to the minInterval. If the
// caller passes a maxInterval less than minInterval, this function will panic.
func NewWorkQueueRunner(name string, fn func(), minInterval, maxInterval time.Duration) *WorkQueueRunner {
	if maxInterval < minInterval {
		panic("maxInterval must be >= minInterval")
	}

	return &WorkQueueRunner{
		name:        name,
		fn:          fn,
		minInterval: minInterval,
		maxInterval: maxInterval,
		queue:       workqueue.NewDelayingQueue(),
		queueKey:    "sync",
		stopCh:      make(chan struct{}),
	}
}

// Run the function as soon as possible. If there is already a queued request to call
// the underlying function, it may be dropped - it is just guaranteed that we will try
// calling the underlying function as soon as possible starting from now.
func (wqr *WorkQueueRunner) Run() {
	wqr.queue.Add(wqr.queueKey)
}

// RetryAfter ensures that the function will run again after no later than interval.
// This can be called from inside a run of the WorkQueueRunner's function, or
// asynchronously.
func (wqr *WorkQueueRunner) RetryAfter(interval time.Duration) {
	wqr.queue.AddAfter(wqr.queueKey, interval)
}

// Loop handles the periodic timer and run requests. This is expected to be
// called as a goroutine.
func (wqr *WorkQueueRunner) Loop(stop <-chan struct{}) {
	klog.V(3).Infof("%s Loop running", wqr.name)

	// Start a goroutine to handle the periodic timer
	go func() {
		ticker := time.NewTicker(wqr.maxInterval)
		defer ticker.Stop()

		for {
			select {
			case <-stop:
				return
			case <-wqr.stopCh:
				return
			case <-ticker.C:
				// Add to the queue to ensure we run at least once per maxInterval
				wqr.queue.Add(wqr.queueKey)
			}
		}
	}()

	// Process the queue
	wait.Until(func() {
		for wqr.processNextWorkItem() {
		}
	}, time.Second, stop)

	wqr.queue.ShutDown()
	close(wqr.stopCh)
}

// processNextWorkItem processes the next item in the queue
func (wqr *WorkQueueRunner) processNextWorkItem() bool {
	key, quit := wqr.queue.Get()
	if quit {
		return false
	}
	defer wqr.queue.Done(key)

	// Check if we need to wait for minInterval
	wqr.lock.Lock()
	elapsed := time.Since(wqr.lastRun)
	if elapsed < wqr.minInterval {
		// Not enough time has passed since the last run
		// Re-queue with a delay
		delay := wqr.minInterval - elapsed
		wqr.lock.Unlock()
		wqr.queue.AddAfter(key, delay)
		klog.V(4).Infof("%s: Too soon to run, requeuing with delay %v", wqr.name, delay)
		return true
	}

	// We can run now
	klog.V(3).Infof("%s: Running function", wqr.name)
	wqr.fn()
	wqr.lastRun = time.Now()
	wqr.lock.Unlock()

	return true
}
