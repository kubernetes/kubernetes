/*
Copyright 2015 The Kubernetes Authors.

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

package scheduler

import (
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog/v2/ktesting"
)

func CheckQueueEq(lhs []string, rhs TimedQueue) bool {
	for i := 0; i < len(lhs); i++ {
		if rhs[i].Value != lhs[i] {
			return false
		}
	}
	return true
}

func CheckSetEq(lhs, rhs sets.Set[string]) bool {
	return lhs.IsSuperset(rhs) && rhs.IsSuperset(lhs)
}

func TestUniqueQueueGet(t *testing.T) {
	var tick int64
	now = func() time.Time {
		t := time.Unix(tick, 0)
		tick++
		return t
	}

	queue := UniqueQueue{
		queue: TimedQueue{},
		set:   sets.New[string](),
	}
	queue.Add(TimedValue{Value: "first", UID: "11111", AddedAt: now(), ProcessAt: now()})
	queue.Add(TimedValue{Value: "second", UID: "22222", AddedAt: now(), ProcessAt: now()})
	queue.Add(TimedValue{Value: "third", UID: "33333", AddedAt: now(), ProcessAt: now()})

	queuePattern := []string{"first", "second", "third"}
	if len(queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have length %d", queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", queue.queue, queuePattern)
	}

	setPattern := sets.New[string]("first", "second", "third")
	if len(queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", queue.set, setPattern)
	}

	queue.Get()
	queuePattern = []string{"second", "third"}
	if len(queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have length %d", queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", queue.queue, queuePattern)
	}

	setPattern = sets.New[string]("second", "third")
	if len(queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", queue.set, setPattern)
	}
}

func TestAddNode(t *testing.T) {
	evictor := NewRateLimitedTimedQueue(flowcontrol.NewFakeAlwaysRateLimiter())
	evictor.Add("first", "11111")
	evictor.Add("second", "22222")
	evictor.Add("third", "33333")

	queuePattern := []string{"first", "second", "third"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have length %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern := sets.New[string]("first", "second", "third")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}
}

func TestDelNode(t *testing.T) {
	defer func() { now = time.Now }()
	var tick int64
	now = func() time.Time {
		t := time.Unix(tick, 0)
		tick++
		return t
	}
	evictor := NewRateLimitedTimedQueue(flowcontrol.NewFakeAlwaysRateLimiter())
	evictor.Add("first", "11111")
	evictor.Add("second", "22222")
	evictor.Add("third", "33333")
	evictor.Remove("first")

	queuePattern := []string{"second", "third"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have length %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern := sets.New[string]("second", "third")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}

	evictor = NewRateLimitedTimedQueue(flowcontrol.NewFakeAlwaysRateLimiter())
	evictor.Add("first", "11111")
	evictor.Add("second", "22222")
	evictor.Add("third", "33333")
	evictor.Remove("second")

	queuePattern = []string{"first", "third"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have length %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern = sets.New[string]("first", "third")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}

	evictor = NewRateLimitedTimedQueue(flowcontrol.NewFakeAlwaysRateLimiter())
	evictor.Add("first", "11111")
	evictor.Add("second", "22222")
	evictor.Add("third", "33333")
	evictor.Remove("third")

	queuePattern = []string{"first", "second"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have length %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern = sets.New[string]("first", "second")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}
}

func TestTry(t *testing.T) {
	evictor := NewRateLimitedTimedQueue(flowcontrol.NewFakeAlwaysRateLimiter())
	evictor.Add("first", "11111")
	evictor.Add("second", "22222")
	evictor.Add("third", "33333")
	evictor.Remove("second")

	deletedMap := sets.New[string]()
	logger, _ := ktesting.NewTestContext(t)
	evictor.Try(logger, func(value TimedValue) (bool, time.Duration) {
		deletedMap.Insert(value.Value)
		return true, 0
	})

	setPattern := sets.New[string]("first", "third")
	if len(deletedMap) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, deletedMap) {
		t.Errorf("Invalid map. Got %v, expected %v", deletedMap, setPattern)
	}
}

func TestTryOrdering(t *testing.T) {
	defer func() { now = time.Now }()
	current := time.Unix(0, 0)
	delay := 0
	// the current time is incremented by 1ms every time now is invoked
	now = func() time.Time {
		if delay > 0 {
			delay--
		} else {
			current = current.Add(time.Millisecond)
		}
		t.Logf("time %d", current.UnixNano())
		return current
	}
	evictor := NewRateLimitedTimedQueue(flowcontrol.NewFakeAlwaysRateLimiter())
	evictor.Add("first", "11111")
	evictor.Add("second", "22222")
	evictor.Add("third", "33333")

	order := []string{}
	count := 0
	hasQueued := false
	logger, _ := ktesting.NewTestContext(t)
	evictor.Try(logger, func(value TimedValue) (bool, time.Duration) {
		count++
		t.Logf("eviction %d", count)
		if value.ProcessAt.IsZero() {
			t.Fatalf("processAt should not be zero")
		}
		switch value.Value {
		case "first":
			if !value.AddedAt.Equal(time.Unix(0, time.Millisecond.Nanoseconds())) {
				t.Fatalf("added time for %s is %v", value.Value, value.AddedAt)
			}

		case "second":
			if !value.AddedAt.Equal(time.Unix(0, 2*time.Millisecond.Nanoseconds())) {
				t.Fatalf("added time for %s is %v", value.Value, value.AddedAt)
			}
			if hasQueued {
				if !value.ProcessAt.Equal(time.Unix(0, 6*time.Millisecond.Nanoseconds())) {
					t.Fatalf("process time for %s is %v", value.Value, value.ProcessAt)
				}
				break
			}
			hasQueued = true
			delay = 1
			t.Logf("going to delay")
			return false, 2 * time.Millisecond

		case "third":
			if !value.AddedAt.Equal(time.Unix(0, 3*time.Millisecond.Nanoseconds())) {
				t.Fatalf("added time for %s is %v", value.Value, value.AddedAt)
			}
		}
		order = append(order, value.Value)
		return true, 0
	})
	if !reflect.DeepEqual(order, []string{"first", "third"}) {
		t.Fatalf("order was wrong: %v", order)
	}
	if count != 3 {
		t.Fatalf("unexpected iterations: %d", count)
	}
}

func TestTryRemovingWhileTry(t *testing.T) {
	evictor := NewRateLimitedTimedQueue(flowcontrol.NewFakeAlwaysRateLimiter())
	evictor.Add("first", "11111")
	evictor.Add("second", "22222")
	evictor.Add("third", "33333")

	processing := make(chan struct{})
	wait := make(chan struct{})
	order := []string{}
	count := 0
	queued := false

	// while the Try function is processing "second", remove it from the queue
	// we should not see "second" retried.
	go func() {
		<-processing
		evictor.Remove("second")
		close(wait)
	}()
	logger, _ := ktesting.NewTestContext(t)
	evictor.Try(logger, func(value TimedValue) (bool, time.Duration) {
		count++
		if value.AddedAt.IsZero() {
			t.Fatalf("added should not be zero")
		}
		if value.ProcessAt.IsZero() {
			t.Fatalf("next should not be zero")
		}
		if !queued && value.Value == "second" {
			queued = true
			close(processing)
			<-wait
			return false, time.Millisecond
		}
		order = append(order, value.Value)
		return true, 0
	})

	if !reflect.DeepEqual(order, []string{"first", "third"}) {
		t.Fatalf("order was wrong: %v", order)
	}
	if count != 3 {
		t.Fatalf("unexpected iterations: %d", count)
	}
}

func TestClear(t *testing.T) {
	evictor := NewRateLimitedTimedQueue(flowcontrol.NewFakeAlwaysRateLimiter())
	evictor.Add("first", "11111")
	evictor.Add("second", "22222")
	evictor.Add("third", "33333")

	evictor.Clear()

	if len(evictor.queue.queue) != 0 {
		t.Fatalf("Clear should remove all elements from the queue.")
	}
}

func TestSwapLimiter(t *testing.T) {
	evictor := NewRateLimitedTimedQueue(flowcontrol.NewFakeAlwaysRateLimiter())
	fakeAlways := flowcontrol.NewFakeAlwaysRateLimiter()
	qps := evictor.limiter.QPS()
	if qps != fakeAlways.QPS() {
		t.Fatalf("QPS does not match create one: %v instead of %v", qps, fakeAlways.QPS())
	}

	evictor.SwapLimiter(0)
	qps = evictor.limiter.QPS()
	fakeNever := flowcontrol.NewFakeNeverRateLimiter()
	if qps != fakeNever.QPS() {
		t.Fatalf("QPS does not match create one: %v instead of %v", qps, fakeNever.QPS())
	}

	createdQPS := float32(5.5)
	evictor.SwapLimiter(createdQPS)
	qps = evictor.limiter.QPS()
	if qps != createdQPS {
		t.Fatalf("QPS does not match create one: %v instead of %v", qps, createdQPS)
	}

	prev := evictor.limiter
	evictor.SwapLimiter(createdQPS)
	if prev != evictor.limiter {
		t.Fatalf("Limiter should not be swapped if the QPS is the same.")
	}
}

func TestAddAfterTry(t *testing.T) {
	evictor := NewRateLimitedTimedQueue(flowcontrol.NewFakeAlwaysRateLimiter())
	evictor.Add("first", "11111")
	evictor.Add("second", "22222")
	evictor.Add("third", "33333")
	evictor.Remove("second")

	deletedMap := sets.New[string]()
	logger, _ := ktesting.NewTestContext(t)
	evictor.Try(logger, func(value TimedValue) (bool, time.Duration) {
		deletedMap.Insert(value.Value)
		return true, 0
	})

	setPattern := sets.New[string]("first", "third")
	if len(deletedMap) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, deletedMap) {
		t.Errorf("Invalid map. Got %v, expected %v", deletedMap, setPattern)
	}

	evictor.Add("first", "11111")
	evictor.Try(logger, func(value TimedValue) (bool, time.Duration) {
		t.Errorf("We shouldn't process the same value if the explicit remove wasn't called.")
		return true, 0
	})
}
