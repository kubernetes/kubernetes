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
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/wait"
)

func TestSimpleQueue(t *testing.T) {
	fakeClock := clock.NewFakeClock(time.Now())
	q := newDelayingQueue(fakeClock, "")

	first := "foo"

	q.AddAfter(first, 50*time.Millisecond)
	if err := waitForWaitingQueueToFill(q); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}

	if q.Len() != 0 {
		t.Errorf("should not have added")
	}

	fakeClock.Step(60 * time.Millisecond)

	if err := waitForAdded(q, 1); err != nil {
		t.Errorf("should have added")
	}
	item, _ := q.Get()
	q.Done(item)

	// step past the next heartbeat
	fakeClock.Step(10 * time.Second)

	err := wait.Poll(1*time.Millisecond, 30*time.Millisecond, func() (done bool, err error) {
		if q.Len() > 0 {
			return false, fmt.Errorf("added to queue")
		}

		return false, nil
	})
	if err != wait.ErrWaitTimeout {
		t.Errorf("expected timeout, got: %v", err)
	}

	if q.Len() != 0 {
		t.Errorf("should not have added")
	}
}

func TestDeduping(t *testing.T) {
	fakeClock := clock.NewFakeClock(time.Now())
	q := newDelayingQueue(fakeClock, "")

	first := "foo"

	q.AddAfter(first, 50*time.Millisecond)
	if err := waitForWaitingQueueToFill(q); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	q.AddAfter(first, 70*time.Millisecond)
	if err := waitForWaitingQueueToFill(q); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if q.Len() != 0 {
		t.Errorf("should not have added")
	}

	// step past the first block, we should receive now
	fakeClock.Step(60 * time.Millisecond)
	if err := waitForAdded(q, 1); err != nil {
		t.Errorf("should have added")
	}
	item, _ := q.Get()
	q.Done(item)

	// step past the second add
	fakeClock.Step(20 * time.Millisecond)
	if q.Len() != 0 {
		t.Errorf("should not have added")
	}

	// test again, but this time the earlier should override
	q.AddAfter(first, 50*time.Millisecond)
	q.AddAfter(first, 30*time.Millisecond)
	if err := waitForWaitingQueueToFill(q); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if q.Len() != 0 {
		t.Errorf("should not have added")
	}

	fakeClock.Step(40 * time.Millisecond)
	if err := waitForAdded(q, 1); err != nil {
		t.Errorf("should have added")
	}
	item, _ = q.Get()
	q.Done(item)

	// step past the second add
	fakeClock.Step(20 * time.Millisecond)
	if q.Len() != 0 {
		t.Errorf("should not have added")
	}
	if q.Len() != 0 {
		t.Errorf("should not have added")
	}
}

func TestAddTwoFireEarly(t *testing.T) {
	fakeClock := clock.NewFakeClock(time.Now())
	q := newDelayingQueue(fakeClock, "")

	first := "foo"
	second := "bar"
	third := "baz"

	q.AddAfter(first, 1*time.Second)
	q.AddAfter(second, 50*time.Millisecond)
	if err := waitForWaitingQueueToFill(q); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}

	if q.Len() != 0 {
		t.Errorf("should not have added")
	}

	fakeClock.Step(60 * time.Millisecond)

	if err := waitForAdded(q, 1); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	item, _ := q.Get()
	if !reflect.DeepEqual(item, second) {
		t.Errorf("expected %v, got %v", second, item)
	}

	q.AddAfter(third, 2*time.Second)

	fakeClock.Step(1 * time.Second)
	if err := waitForAdded(q, 1); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	item, _ = q.Get()
	if !reflect.DeepEqual(item, first) {
		t.Errorf("expected %v, got %v", first, item)
	}

	fakeClock.Step(2 * time.Second)
	if err := waitForAdded(q, 1); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	item, _ = q.Get()
	if !reflect.DeepEqual(item, third) {
		t.Errorf("expected %v, got %v", third, item)
	}
}

func TestCopyShifting(t *testing.T) {
	fakeClock := clock.NewFakeClock(time.Now())
	q := newDelayingQueue(fakeClock, "")

	first := "foo"
	second := "bar"
	third := "baz"

	q.AddAfter(first, 1*time.Second)
	q.AddAfter(second, 500*time.Millisecond)
	q.AddAfter(third, 250*time.Millisecond)
	if err := waitForWaitingQueueToFill(q); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}

	if q.Len() != 0 {
		t.Errorf("should not have added")
	}

	fakeClock.Step(2 * time.Second)

	if err := waitForAdded(q, 3); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
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
}

func BenchmarkDelayingQueue_AddAfter(b *testing.B) {
	fakeClock := clock.NewFakeClock(time.Now())
	q := newDelayingQueue(fakeClock, "")

	// Add items
	for n := 0; n < b.N; n++ {
		data := fmt.Sprintf("%d", n)
		q.AddAfter(data, time.Duration(rand.Int63n(int64(10*time.Minute))))
	}

	// Exercise item removal as well
	fakeClock.Step(11 * time.Minute)
	for n := 0; n < b.N; n++ {
		_, _ = q.Get()
	}
}

func waitForAdded(q DelayingInterface, depth int) error {
	return wait.Poll(1*time.Millisecond, 10*time.Second, func() (done bool, err error) {
		if q.Len() == depth {
			return true, nil
		}

		return false, nil
	})
}

func waitForWaitingQueueToFill(q DelayingInterface) error {
	return wait.Poll(1*time.Millisecond, 10*time.Second, func() (done bool, err error) {
		if len(q.(*delayingType).waitingForAddCh) == 0 {
			return true, nil
		}

		return false, nil
	})
}
