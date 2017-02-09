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

	"k8s.io/client-go/util/clock"
)

func TestRateLimitingQueue(t *testing.T) {
	limiter := NewItemExponentialFailureRateLimiter(1*time.Millisecond, 1*time.Second)
	queue := NewRateLimitingQueue(limiter).(*rateLimitingType)
	fakeClock := clock.NewFakeClock(time.Now())
	delayingQueue := &delayingType{
		Interface:       New(),
		clock:           fakeClock,
		heartbeat:       fakeClock.Tick(maxWait),
		stopCh:          make(chan struct{}),
		waitingForAddCh: make(chan waitFor, 1000),
		metrics:         newRetryMetrics(""),
	}
	queue.DelayingInterface = delayingQueue

	queue.AddRateLimited("one")
	waitEntry := <-delayingQueue.waitingForAddCh
	if e, a := 1*time.Millisecond, waitEntry.readyAt.Sub(fakeClock.Now()); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	queue.AddRateLimited("one")
	waitEntry = <-delayingQueue.waitingForAddCh
	if e, a := 2*time.Millisecond, waitEntry.readyAt.Sub(fakeClock.Now()); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := 2, queue.NumRequeues("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	queue.AddRateLimited("two")
	waitEntry = <-delayingQueue.waitingForAddCh
	if e, a := 1*time.Millisecond, waitEntry.readyAt.Sub(fakeClock.Now()); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	queue.AddRateLimited("two")
	waitEntry = <-delayingQueue.waitingForAddCh
	if e, a := 2*time.Millisecond, waitEntry.readyAt.Sub(fakeClock.Now()); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	queue.Forget("one")
	if e, a := 0, queue.NumRequeues("one"); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	queue.AddRateLimited("one")
	waitEntry = <-delayingQueue.waitingForAddCh
	if e, a := 1*time.Millisecond, waitEntry.readyAt.Sub(fakeClock.Now()); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

}
