/*
Copyright 2020 The Kubernetes Authors.

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

package endpoint

import (
	"testing"
	"time"

	"k8s.io/klog"

	"golang.org/x/time/rate"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
)

func waitForQueueNonEmpty(queue *DeduplicatingRateLimitingQueue) error {
	return wait.PollImmediate(time.Millisecond, 100*time.Millisecond, func() (bool, error) {
		return queue.Len() > 0, nil
	})
}

func expectKey(t *testing.T, queue *DeduplicatingRateLimitingQueue, want string) {
	if err := waitForQueueNonEmpty(queue); err != nil {
		t.Fatalf("expected at least one element: %v", err)
	}

	if got, _ := queue.Get(); got != want {
		t.Fatalf("queue.Get() = %v; want: %v", got, want)
	}
	queue.Done(want)
	queue.Forget(want)
}

func TestBasic(t *testing.T) {
	clock := clock.NewFakeClock(time.Now())
	queue := NewDeduplicatingRateLimitingQueue(
		workqueue.NewNamedRateLimitingQueueWithCustomClock(workqueue.DefaultControllerRateLimiter(), "", clock),
		// There is no way to mock clock in NewItemBucketRateLimiter,
		// so we are not able to test a case when we are stopped being rate limited.
		workqueue.NewItemBucketRateLimiter(rate.Limit(1.0), 1),
	)
	defer queue.ShutDown()

	// First call without delay.
	queue.Enqueue("key1", true)

	expectKey(t, queue, "key1")

	// key1 should be rate limited now
	queue.Enqueue("key1", true)
	queue.Enqueue("key2", true)

	// key1 is added with 1s delay, so key2 is only available key now.
	expectKey(t, queue, "key2")

	clock.Step(1 * time.Second)
	expectKey(t, queue, "key1")
}

func TestDedup(t *testing.T) {
	clock := clock.NewFakeClock(time.Now())
	queue := NewDeduplicatingRateLimitingQueue(
		workqueue.NewNamedRateLimitingQueueWithCustomClock(workqueue.DefaultControllerRateLimiter(), "", clock),
		// There is no way to mock clock in NewItemBucketRateLimiter,
		// so we are not able to test a case when we are stopped being rate limited.
		workqueue.NewItemBucketRateLimiter(rate.Limit(1.0), 1),
	)
	defer queue.ShutDown()

	// We add key1 three times. The expected result is that it will be deduped.
	queue.Enqueue("key1", false)
	queue.Enqueue("key1", false)
	queue.Enqueue("key1", false)

	queue.Enqueue("key2", false)

	expectKey(t, queue, "key1")
	expectKey(t, queue, "key2")
}

func TestAllowRateLimited(t *testing.T) {
	clock := clock.NewFakeClock(time.Now())
	queue := NewDeduplicatingRateLimitingQueue(
		workqueue.NewNamedRateLimitingQueueWithCustomClock(workqueue.DefaultControllerRateLimiter(), "", clock),
		// There is no way to mock clock in NewItemBucketRateLimiter,
		// so we are not able to test a case when we are stopped being rate limited.
		workqueue.NewItemBucketRateLimiter(rate.Limit(1.0), 1),
	)
	defer queue.ShutDown()

	// First call without delay.
	queue.Enqueue("key1", true)

	expectKey(t, queue, "key1")

	// key1 should be rate limited now
	queue.Enqueue("key1", false)
	queue.Enqueue("key2", true)

	// key1 has been added with allowRateLimit false, so it's not rate limited.
	expectKey(t, queue, "key1")
	expectKey(t, queue, "key2")
}

func TestEnqueueRetry(t *testing.T) {
	maxDelay := 15 * time.Minute

	clock := clock.NewFakeClock(time.Now())
	queue := NewDeduplicatingRateLimitingQueue(
		workqueue.NewNamedRateLimitingQueueWithCustomClock(workqueue.NewItemExponentialFailureRateLimiter(5*time.Millisecond, maxDelay), "", clock),
		// There is no way to mock clock in NewItemBucketRateLimiter,
		// so we are not able to test a case when we are stopped being rate limited.
		workqueue.NewItemBucketRateLimiter(rate.Limit(1.0), 1),
	)
	defer queue.ShutDown()

	// First call without delay.
	queue.Enqueue("key1", true)
	expectKey(t, queue, "key1")

	// Simulate series of failures
	for i := 0; i < 100; i++ {
		queue.EnqueueRetry("key1")
		clock.Step(maxDelay)
		klog.Infof("Waiting for %v", i)
		key, _ := queue.Get()
		if key != "key1" {
			t.Fatalf("queue.Get() = %v; want key1", key)
		}
		queue.Done("key1")
	}

	queue.Enqueue("key2", true)
	// We expect key2 now as key1 waits for maxDelay now.
	expectKey(t, queue, "key2")

	// Next queue.Enqueue should shorten delay for key1 to 1 second (delay from standard 1 qps rate limiter).
	queue.Enqueue("key1", true)
	clock.Step(1 * time.Second)
	expectKey(t, queue, "key1")
}
