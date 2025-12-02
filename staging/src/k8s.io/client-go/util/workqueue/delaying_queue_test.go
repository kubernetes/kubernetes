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
	"fmt"
	"math/rand"
	"reflect"
	"testing"
	"testing/synctest"
	"time"

	testingclock "k8s.io/utils/clock/testing"
)

func TestSimpleQueue(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := NewDelayingQueue()
		defer q.ShutDown()

		first := "foo"

		q.AddAfter(first, 50*time.Millisecond)
		synctest.Wait() // wait for waiting queue to fill

		if q.Len() != 0 {
			t.Errorf("should not have added")
		}

		time.Sleep(60 * time.Millisecond)

		waitForAdded(t, q, 1)
		item, _ := q.Get()
		q.Done(item)

		// step past the next heartbeat
		time.Sleep(10 * time.Second)

		if q.Len() != 0 {
			t.Errorf("should not have added")
		}
	})
}

func TestDeduping(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		mp := testMetricsProvider{}
		q := NewDelayingQueueWithConfig(DelayingQueueConfig{
			Name:            "test-delay",
			MetricsProvider: &mp,
		})
		defer q.ShutDown()

		first := "foo"

		q.AddAfter(first, 50*time.Millisecond)
		synctest.Wait() // wait for waiting queue to fill
		if mp.delayed.gaugeValue() != 1 {
			t.Errorf("expected 1 delayed, got %v", mp.delayed.gaugeValue())
		}
		if mp.retries.gaugeValue() != 1 {
			t.Errorf("expected 1 retry, got %v", mp.retries.gaugeValue())
		}

		q.AddAfter(first, 70*time.Millisecond)
		synctest.Wait() // wait for waiting queue to fill
		if q.Len() != 0 {
			t.Errorf("should not have added")
		}
		if mp.delayed.gaugeValue() != 1 {
			t.Errorf("expected 1 delayed, got %v", mp.delayed.gaugeValue())
		}
		if mp.retries.gaugeValue() != 2 {
			t.Errorf("expected 2 retries, got %v", mp.retries.gaugeValue())
		}

		// step past the first block, we should receive now
		time.Sleep(60 * time.Millisecond)
		waitForAdded(t, q, 1)
		if mp.delayed.gaugeValue() != 0 {
			t.Errorf("expected 0 delayed, got %v", mp.delayed.gaugeValue())
		}
		item, _ := q.Get()
		q.Done(item)

		// step past the second add
		time.Sleep(20 * time.Millisecond)
		if q.Len() != 0 {
			t.Errorf("should not have added")
		}

		// test again, but this time the earlier should override
		q.AddAfter(first, 50*time.Millisecond)
		q.AddAfter(first, 30*time.Millisecond)
		synctest.Wait() // wait for waiting queue to fill
		if q.Len() != 0 {
			t.Errorf("should not have added")
		}
		if mp.delayed.gaugeValue() != 1 {
			t.Errorf("expected 1 delayed, got %v", mp.delayed.gaugeValue())
		}
		if mp.retries.gaugeValue() != 4 {
			t.Errorf("expected 4 retries, got %v", mp.retries.gaugeValue())
		}

		time.Sleep(40 * time.Millisecond)
		waitForAdded(t, q, 1)
		if mp.delayed.gaugeValue() != 0 {
			t.Errorf("expected 0 delayed, got %v", mp.delayed.gaugeValue())
		}
		item, _ = q.Get()
		q.Done(item)

		// step past the second add
		time.Sleep(20 * time.Millisecond)
		if q.Len() != 0 {
			t.Errorf("should not have added")
		}
	})
}

func TestAddTwoFireEarly(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := NewDelayingQueue()
		defer q.ShutDown()

		first := "foo"
		second := "bar"
		third := "baz"

		q.AddAfter(first, 1*time.Second)
		q.AddAfter(second, 50*time.Millisecond)
		synctest.Wait() // wait for waiting queue to fill

		if q.Len() != 0 {
			t.Errorf("should not have added")
		}

		time.Sleep(60 * time.Millisecond)

		waitForAdded(t, q, 1)
		item, _ := q.Get()
		if !reflect.DeepEqual(item, second) {
			t.Errorf("expected %v, got %v", second, item)
		}

		q.AddAfter(third, 2*time.Second)

		time.Sleep(1 * time.Second)
		waitForAdded(t, q, 1)
		item, _ = q.Get()
		if !reflect.DeepEqual(item, first) {
			t.Errorf("expected %v, got %v", first, item)
		}

		time.Sleep(2 * time.Second)
		waitForAdded(t, q, 1)
		item, _ = q.Get()
		if !reflect.DeepEqual(item, third) {
			t.Errorf("expected %v, got %v", third, item)
		}
	})
}

func TestCopyShifting(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := NewDelayingQueue()
		defer q.ShutDown()

		first := "foo"
		second := "bar"
		third := "baz"

		q.AddAfter(first, 1*time.Second)
		q.AddAfter(second, 500*time.Millisecond)
		q.AddAfter(third, 250*time.Millisecond)
		synctest.Wait() // wait for waiting queue to fill

		if q.Len() != 0 {
			t.Errorf("should not have added")
		}

		time.Sleep(2 * time.Second)

		waitForAdded(t, q, 3)
		actualFirst, _ := q.Get()
		if !reflect.DeepEqual(actualFirst, third) {
			t.Errorf("expected %v, got %v", third, actualFirst)
		}
		actualSecond, _ := q.Get()
		if !reflect.DeepEqual(actualSecond, second) {
			t.Errorf("expected %v, got %v", second, actualSecond)
		}
		actualThird, _ := q.Get()
		if !reflect.DeepEqual(actualThird, first) {
			t.Errorf("expected %v, got %v", first, actualThird)
		}
	})
}

func BenchmarkDelayingQueue_AddAfter(b *testing.B) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	q := NewTypedDelayingQueueWithConfig(TypedDelayingQueueConfig[string]{Clock: fakeClock})

	// Add items
	for n := range b.N {
		data := fmt.Sprintf("%d", n)
		q.AddAfter(data, time.Duration(rand.Int63n(int64(10*time.Minute))))
	}

	// Exercise item removal as well
	fakeClock.Step(11 * time.Minute)
	for range b.N {
		_, _ = q.Get()
	}
}

func waitForAdded(t *testing.T, q DelayingInterface, depth int) {
	t.Helper()
	synctest.Wait()
	if q.Len() != depth {
		t.Fatalf("expected depth %d, got %d", depth, q.Len())
	}
}
