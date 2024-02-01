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

	testingclock "k8s.io/utils/clock/testing"
)

// TestRateLimitingQueueLayer tests the implementation of the
// extra behavior beyond the delaying type.
func TestRateLimitingQueueLayer(t *testing.T) {
	limiter := NewItemExponentialFailureRateLimiter(1*time.Millisecond, 1*time.Second)
	fakeClock := testingclock.NewFakeClock(time.Now())
	fifo := NewWithConfig(QueueConfig{Name: "test", Clock: fakeClock})
	delayingQueue := createDelayingQueue(fakeClock, fifo, "test", nil)
	queue := NewRateLimitingQueueWithConfig(limiter, RateLimitingQueueConfig{
		Name:          "test",
		Clock:         fakeClock,
		DelayingQueue: delayingQueue,
	}).(*rateLimitingType)

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

func TestRateLimitingQueueBehavior(t *testing.T) {
	limiter := NewItemExponentialFailureRateLimiter(1*time.Millisecond, 1*time.Second)
	// loc, _ := time.LoadLocation("UTC")
	// startTime := time.Date(2024, 1, 1, 1, 0, 0, 0, loc)
	startTime := time.Now()
	fakeClock := testingclock.NewFakeClock(startTime)
	queue := NewRateLimitingQueueWithConfig(limiter, RateLimitingQueueConfig{
		Clock: fakeClock,
	})
	queue.Add("one")
	queue.AddRateLimited("one")
	fakeClock.Step(2 * time.Millisecond)
	time.Sleep(5 * time.Second)
	if fakeClock.Since(startTime) != 2*time.Millisecond {
		t.Fatal("Wrong time")
	}
	if expected, actual := 1, queue.Len(); expected != actual {
		t.Fatalf("Expected Len %d, got %d", expected, actual)
	}
	item, _ := queue.Get()
	queue.Done(item)
	if item != "one" {
		t.Fatalf("Expected %#v, got %#v", "one", item)
	}
	if expected, actual := 0, queue.Len(); expected != actual {
		t.Fatalf("Expected Len %d, got %d", expected, actual)
	}
	// Expect the AddRateLimited above did not call the rate limiter.
	// Expect only the first call below calls the rate limiter.
	queue.AddRateLimited("one")
	time.Sleep(5 * time.Second)
	queue.AddRateLimited("one")
	time.Sleep(5 * time.Second)
	queue.AddRateLimited("one")
	fakeClock.Step(1 * time.Millisecond)
	time.Sleep(5 * time.Second)
	if fakeClock.Since(startTime) != 3*time.Millisecond {
		t.Fatal("Wrong time")
	}
	if expected, actual := 1, queue.Len(); expected != actual {
		t.Fatalf("Expected Len %d, got %d", expected, actual)
	}
	fakeClock.Step(1 * time.Second)
	time.Sleep(5 * time.Second)
	if fakeClock.Since(startTime) != 1003*time.Millisecond {
		t.Fatal("Wrong time")
	}
	if expected, actual := 1, queue.Len(); expected != actual {
		t.Fatalf("Expected Len %d, got %d", expected, actual)
	}
	item, _ = queue.Get()
	queue.Done(item)
	if item != "one" {
		t.Fatalf("Expected %#v, got %#v", "one", item)
	}
	// Expect one more call to the rate limiter, now yielding 2 ms
	queue.AddRateLimited("one")
	time.Sleep(5 * time.Second)
	queue.AddRateLimited("one")
	time.Sleep(5 * time.Second)
	queue.AddRateLimited("one")
	fakeClock.Step(1 * time.Millisecond)
	time.Sleep(5 * time.Second)
	if fakeClock.Since(startTime) != 1004*time.Millisecond {
		t.Fatal("Wrong time")
	}
	if expected, actual := 0, queue.Len(); expected != actual {
		t.Fatalf("Expected Len %d, got %d", expected, actual)
	}
	fakeClock.Step(1 * time.Millisecond)
	time.Sleep(5 * time.Second)
	if fakeClock.Since(startTime) != 1005*time.Millisecond {
		t.Fatal("Wrong time")
	}
	if expected, actual := 1, queue.Len(); expected != actual {
		t.Fatalf("Expected Len %d, got %d", expected, actual)
	}
	fakeClock.Step(1 * time.Second)
	time.Sleep(5 * time.Second)
	if fakeClock.Since(startTime) != 2005*time.Millisecond {
		t.Fatal("Wrong time")
	}
	if expected, actual := 1, queue.Len(); expected != actual {
		t.Fatalf("Expected Len %d, got %d", expected, actual)
	}
	item, _ = queue.Get()
	queue.Done(item)
	if item != "one" {
		t.Fatalf("Expected %#v, got %#v", "one", item)
	}
}
