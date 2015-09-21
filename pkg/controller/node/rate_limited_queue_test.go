/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package nodecontroller

import (
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

func CheckQueueEq(lhs []string, rhs TimedQueue) bool {
	for i := 0; i < len(lhs); i++ {
		if rhs[i].Value != lhs[i] {
			return false
		}
	}
	return true
}

func CheckSetEq(lhs, rhs sets.String) bool {
	return lhs.HasAll(rhs.List()...) && rhs.HasAll(lhs.List()...)
}

func TestAddNode(t *testing.T) {
	evictor := NewRateLimitedTimedQueue(util.NewFakeRateLimiter())
	evictor.Add("first")
	evictor.Add("second")
	evictor.Add("third")

	queuePattern := []string{"first", "second", "third"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have length %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern := sets.NewString("first", "second", "third")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}
}

func TestDelNode(t *testing.T) {
	evictor := NewRateLimitedTimedQueue(util.NewFakeRateLimiter())
	evictor.Add("first")
	evictor.Add("second")
	evictor.Add("third")
	evictor.Remove("first")

	queuePattern := []string{"second", "third"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have length %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern := sets.NewString("second", "third")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}

	evictor = NewRateLimitedTimedQueue(util.NewFakeRateLimiter())
	evictor.Add("first")
	evictor.Add("second")
	evictor.Add("third")
	evictor.Remove("second")

	queuePattern = []string{"first", "third"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have length %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern = sets.NewString("first", "third")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}

	evictor = NewRateLimitedTimedQueue(util.NewFakeRateLimiter())
	evictor.Add("first")
	evictor.Add("second")
	evictor.Add("third")
	evictor.Remove("third")

	queuePattern = []string{"first", "second"}
	if len(evictor.queue.queue) != len(queuePattern) {
		t.Fatalf("Queue %v should have length %d", evictor.queue.queue, len(queuePattern))
	}
	if !CheckQueueEq(queuePattern, evictor.queue.queue) {
		t.Errorf("Invalid queue. Got %v, expected %v", evictor.queue.queue, queuePattern)
	}

	setPattern = sets.NewString("first", "second")
	if len(evictor.queue.set) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, evictor.queue.set) {
		t.Errorf("Invalid map. Got %v, expected %v", evictor.queue.set, setPattern)
	}
}

func TestTry(t *testing.T) {
	evictor := NewRateLimitedTimedQueue(util.NewFakeRateLimiter())
	evictor.Add("first")
	evictor.Add("second")
	evictor.Add("third")
	evictor.Remove("second")

	deletedMap := sets.NewString()
	evictor.Try(func(value TimedValue) (bool, time.Duration) {
		deletedMap.Insert(value.Value)
		return true, 0
	})

	setPattern := sets.NewString("first", "third")
	if len(deletedMap) != len(setPattern) {
		t.Fatalf("Map %v should have length %d", evictor.queue.set, len(setPattern))
	}
	if !CheckSetEq(setPattern, deletedMap) {
		t.Errorf("Invalid map. Got %v, expected %v", deletedMap, setPattern)
	}
}

func TestTryOrdering(t *testing.T) {
	evictor := NewRateLimitedTimedQueue(util.NewFakeRateLimiter())
	evictor.Add("first")
	evictor.Add("second")
	evictor.Add("third")

	order := []string{}
	count := 0
	queued := false
	evictor.Try(func(value TimedValue) (bool, time.Duration) {
		count++
		if value.AddedAt.IsZero() {
			t.Fatalf("added should not be zero")
		}
		if value.ProcessAt.IsZero() {
			t.Fatalf("next should not be zero")
		}
		if !queued && value.Value == "second" {
			queued = true
			return false, time.Millisecond
		}
		order = append(order, value.Value)
		return true, 0
	})
	if !reflect.DeepEqual(order, []string{"first", "third", "second"}) {
		t.Fatalf("order was wrong: %v", order)
	}
	if count != 4 {
		t.Fatalf("unexpected iterations: %d", count)
	}
}

func TestTryRemovingWhileTry(t *testing.T) {
	evictor := NewRateLimitedTimedQueue(util.NewFakeRateLimiter())
	evictor.Add("first")
	evictor.Add("second")
	evictor.Add("third")

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

	evictor.Try(func(value TimedValue) (bool, time.Duration) {
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
