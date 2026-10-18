/*
Copyright The Kubernetes Authors.

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

package podautoscaler

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
	testingclock "k8s.io/utils/clock/testing"
)

func TestPerItemIntervalRateLimiter(t *testing.T) {
	rateLimiter := NewPerItemIntervalRateLimiter(15 * time.Second)

	assert.Equal(t, 15*time.Second, rateLimiter.When("ns/a"), "items without an override use the default interval")

	rateLimiter.SetInterval("ns/a", 5*time.Second)
	interval, override := rateLimiter.Interval("ns/a")
	assert.Equal(t, 5*time.Second, interval)
	assert.True(t, override)
	assert.Equal(t, 5*time.Second, rateLimiter.When("ns/a"))

	interval, override = rateLimiter.Interval("ns/b")
	assert.Equal(t, 15*time.Second, interval, "other items keep the default interval")
	assert.False(t, override)

	rateLimiter.Forget("ns/a")
	assert.Equal(t, 5*time.Second, rateLimiter.When("ns/a"), "Forget keeps the override")
	assert.Equal(t, 1, rateLimiter.NumRequeues("ns/a"))

	rateLimiter.RemoveInterval("ns/a")
	interval, override = rateLimiter.Interval("ns/a")
	assert.Equal(t, 15*time.Second, interval, "RemoveInterval restores the default interval")
	assert.False(t, override)
}

func TestPerItemIntervalRateLimiterQueueTiming(t *testing.T) {
	const key = "test-ns/test-hpa"

	t.Run("uses the per-item interval", func(t *testing.T) {
		rateLimiter := NewPerItemIntervalRateLimiter(15 * time.Second)
		rateLimiter.SetInterval(key, 5*time.Second)
		clk, queue := newFakeClockQueue(t, rateLimiter)

		queue.AddRateLimited(key)
		clk.Step(4 * time.Second)
		assert.Equal(t, 0, queue.Len(), "the item must not be ready before its interval")
		clk.Step(time.Second)
		waitForQueueLen(t, queue, 1)
	})

	t.Run("uses the default interval without an override", func(t *testing.T) {
		rateLimiter := NewPerItemIntervalRateLimiter(15 * time.Second)
		clk, queue := newFakeClockQueue(t, rateLimiter)

		queue.AddRateLimited(key)
		clk.Step(14 * time.Second)
		assert.Equal(t, 0, queue.Len(), "the item must not be ready before the default interval")
		clk.Step(time.Second)
		waitForQueueLen(t, queue, 1)
	})

	t.Run("a shorter interval moves the waiting item earlier", func(t *testing.T) {
		rateLimiter := NewPerItemIntervalRateLimiter(15 * time.Second)
		clk, queue := newFakeClockQueue(t, rateLimiter)

		queue.AddRateLimited(key)
		rateLimiter.SetInterval(key, 5*time.Second)
		queue.AddRateLimited(key)
		clk.Step(5 * time.Second)
		waitForQueueLen(t, queue, 1)
	})

	t.Run("a longer interval applies after the waiting item runs", func(t *testing.T) {
		rateLimiter := NewPerItemIntervalRateLimiter(15 * time.Second)
		rateLimiter.SetInterval(key, 5*time.Second)
		clk, queue := newFakeClockQueue(t, rateLimiter)

		queue.AddRateLimited(key)
		rateLimiter.SetInterval(key, 30*time.Second)
		queue.AddRateLimited(key)
		flushWaitingItems(t, clk, queue)

		clk.Step(5 * time.Second)
		waitForQueueLen(t, queue, 1)
		item, _ := queue.Get()
		queue.Done(item)

		queue.AddRateLimited(key)
		clk.Step(29 * time.Second)
		assert.Equal(t, 0, queue.Len(), "the item must not be ready before the new interval")
		clk.Step(time.Second)
		waitForQueueLen(t, queue, 1)
	})
}

func newFakeClockQueue(t *testing.T, rateLimiter workqueue.TypedRateLimiter[string]) (*testingclock.FakeClock, workqueue.TypedRateLimitingInterface[string]) {
	t.Helper()
	clk := testingclock.NewFakeClock(time.Now())
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(rateLimiter, workqueue.TypedRateLimitingQueueConfig[string]{Clock: clk})
	t.Cleanup(queue.ShutDown)
	return clk, queue
}

func waitForQueueLen(t *testing.T, queue workqueue.TypedRateLimitingInterface[string], length int) {
	t.Helper()
	err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		return queue.Len() == length, nil
	})
	require.NoError(t, err, "queue length did not become %d, it is %d", length, queue.Len())
}

// flushWaitingItems returns once the delaying queue has processed all earlier
// AddAfter calls, so that stepping the fake clock afterwards cannot race with
// them. It advances the fake clock by one nanosecond.
func flushWaitingItems(t *testing.T, clk *testingclock.FakeClock, queue workqueue.TypedRateLimitingInterface[string]) {
	t.Helper()
	const marker = "flush-marker"
	queue.AddAfter(marker, time.Nanosecond)
	clk.Step(time.Nanosecond)
	waitForQueueLen(t, queue, 1)
	item, _ := queue.Get()
	require.Equal(t, marker, item)
	queue.Done(item)
}
