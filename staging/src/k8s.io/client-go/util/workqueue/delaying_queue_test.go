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
	"context"
	"fmt"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	testingclock "k8s.io/utils/clock/testing"
)

const (
	DefaultWaitTimeout = 2 * time.Second      // The default duration we will wait before giving up when polling for a result.
	DefaultWaitStep    = 5 * time.Millisecond // The default pause between attempted polls when polling for a result.
)

func TestSimpleQueue(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	q := NewDelayingQueueWithConfig(DelayingQueueConfig{Clock: fakeClock})

	first := "foo"

	q.AddAfter(first, 50*time.Millisecond)
	q.syncFill()

	if q.Len() != 0 {
		t.Errorf("should not have added")
	}

	fakeClock.Step(60 * time.Millisecond)

	if err := waitForActiveQueueDepth(q, 1); err != nil {
		t.Errorf("should have added")
	}
	item, _ := q.Get()
	q.Done(item)

	// step past the next heartbeat
	fakeClock.Step(10 * time.Second)

	ctx := context.TODO()
	err := wait.PollUntilContextTimeout(ctx, DefaultWaitStep, DefaultWaitTimeout, true, func(ctx context.Context) (done bool, err error) {
		if q.Len() > 0 {
			return false, fmt.Errorf("added to queue")
		}
		return false, nil
	})
	if !wait.Interrupted(err) {
		t.Errorf("expected timeout, got: %v", err)
	}

	if q.Len() != 0 {
		t.Errorf("should not have added")
	}
}

// TestImmediateAdd calls to Add() should 'promote' items out of waiting (i.e. delete from waiting)
// and immediately add them (via the waitingLoop)
func TestImmediateAdd(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now().Truncate(1 * time.Second))
	q := NewDelayingQueueWithCustomClock(fakeClock, "")

	first := "foo"
	second := "foo"

	q.AddAfter(first, 1*time.Second)
	q.syncFill()
	assertDelayedQueueState(t, q, QueueState{
		waiting: []interface{}{first},
	})
	// the immediate add leaves the waiting item in the queue
	q.Add(second)
	assertDelayedQueueState(t, q, QueueState{
		active:  []interface{}{second},
		waiting: []interface{}{first},
	})
	item, _ := q.Get()
	q.Done(item)

	// The first item is still in the waiting queue
	assertDelayedQueueState(t, q, QueueState{
		waiting: []interface{}{first},
	})

	// Add an immediate add with AddWithOptions will remove the waiting item
	q.AddWithOptions(second, ExpandedDelayingOptions{})
	q.syncFill()
	assertDelayedQueueState(t, q, QueueState{
		active: []interface{}{first},
	})

	// AddAfter should always queue a waiting item when it is already in the "active" queue
	q.AddAfter(first, 1*time.Second)
	q.syncFill()
	assertDelayedQueueState(t, q, QueueState{
		active:  []interface{}{first},
		waiting: []interface{}{first},
	})
	// remove the active "foo"
	item, _ = q.Get()
	q.Done(item)

	// Show that AddWithOptions with the PermitActiveAndWaiting option gives us the original behaviour again
	q.AddWithOptions(second, ExpandedDelayingOptions{PermitActiveAndWaiting: true})
	q.syncFill()
	assertDelayedQueueState(t, q, QueueState{
		active:  []interface{}{second},
		waiting: []interface{}{first},
	})
	item, _ = q.Get()
	q.Done(item)
}

// TestOptionsTakeLongerTakeExisting - these options give the client an ability to perform a conditional add but
// only when the item isn't already queued.  This functionality also applies to immediate adds made by the AddWithOptions
// method.
func TestOptionsTakeLongerTakeExisting(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now().Truncate(1 * time.Second))
	q := NewDelayingQueueWithCustomClock(fakeClock, "")

	first := "foo"
	second := "foo"
	third := "bar"

	// add an item into the waiting queue.
	q.AddAfter(first, 2*time.Second)

	// attempt to immediately add an item using TakeLonger option
	q.AddWithOptions(second, ExpandedDelayingOptions{WhenWaiting: TakeLonger})
	assertDelayedQueueState(t, q, QueueState{
		waiting: []interface{}{first},
	})

	// attempt to immediately add an item using TakeExisting option
	q.AddWithOptions(second, ExpandedDelayingOptions{WhenWaiting: TakeExisting})
	assertDelayedQueueState(t, q, QueueState{
		waiting: []interface{}{first},
	})

	// attempt to add an item not queued should queue successfully...
	q.AddWithOptions(third, ExpandedDelayingOptions{WhenWaiting: TakeExisting})
	assertDelayedQueueState(t, q, QueueState{
		active:  []interface{}{third},
		waiting: []interface{}{first},
	})

	// step past the original delayed item and process it from the active queue
	fakeClock.Step(5 * time.Second)
	if err := waitForActiveQueueDepth(q, 2); err != nil {
		t.Fatalf("the first item should have been queued")
	}
	assertDelayedQueueState(t, q, QueueState{
		active: []interface{}{first, third},
	})
	// pop them both off the queue
	item, _ := q.Get()
	q.Done(item)
	item, _ = q.Get()
	q.Done(item)

	// add the second item again with DropIfWaiting which should now succeed
	q.AddWithOptions(second, ExpandedDelayingOptions{WhenWaiting: TakeExisting})
	assertDelayedQueueState(t, q, QueueState{
		active: []interface{}{second},
	})
}

// TestDelayedInsertTakeShorter - tests the insert behaviour with this WhenWaiting behaviour (the default)
func TestDelayedInsertTakeShorter(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now().Truncate(1 * time.Second))
	q := NewDelayingQueueWithCustomClock(fakeClock, "")

	first := "baz"
	second := "foo"
	third := "foo"

	// add an item into the waiting queue.
	q.AddAfter(first, 10*time.Second)
	q.AddAfter(second, 10*time.Second)
	// attempt add delayed again with a shorter time time out and TakeShorter option.
	q.AddWithOptions(third, ExpandedDelayingOptions{Duration: 1 * time.Second, WhenWaiting: TakeShorter})
	q.syncFill()
	// step past the shorter item and confirm that it got queued.
	fakeClock.Step(2 * time.Second)
	if err := waitForActiveQueueDepth(q, 1); err != nil {
		t.Fatalf("expected success, got: %v", err)
	}
	// we should be left with baz in waiting and foo should now be active
	assertDelayedQueueState(t, q, QueueState{
		active:  []interface{}{third},
		waiting: []interface{}{first},
	})
}

// TestDelayedInsertTakeLonger- tests the insert behaviour with this WhenWaiting behaviour.
func TestDelayedInsertTakeLonger(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now().Truncate(1 * time.Second))
	q := NewDelayingQueueWithCustomClock(fakeClock, "")

	first := "baz"
	second := "foo"
	third := "foo"
	fourth := "foo"

	// add an item into the waiting queue.
	q.AddAfter(first, 60*time.Second)
	q.AddAfter(second, 5*time.Second)
	// attempt add delayed again with a shorter time time out and TakeLonger option.
	q.AddWithOptions(third, ExpandedDelayingOptions{Duration: 1 * time.Second, WhenWaiting: TakeLonger})
	q.syncFill()
	// step past the shorter item and confirm that it didn't get queued
	fakeClock.Step(2 * time.Second)
	if err := waitForActiveQueueDepth(q, 1); !wait.Interrupted(err) {
		t.Fatalf("expected timeout, got: %v", err)
	}
	// Add a longer delay with the expectation of pushing the ready later...
	q.AddWithOptions(fourth, ExpandedDelayingOptions{Duration: 20 * time.Second, WhenWaiting: TakeLonger})
	q.syncFill()
	// step past the original delayed item and check it queues.
	fakeClock.Step(4 * time.Second)
	if err := waitForActiveQueueDepth(q, 1); !wait.Interrupted(err) {
		t.Fatalf("expected timeout, got: %v", err)
	}
	// step past the new longer delay and prove our longer item was queued.
	fakeClock.Step(30 * time.Second)
	if err := waitForActiveQueueDepth(q, 1); err != nil {
		t.Fatalf("expected success, got: %v", err)
	}
	// we should be left with baz in waiting and foo should now be active
	assertDelayedQueueState(t, q, QueueState{
		active:  []interface{}{third},
		waiting: []interface{}{first},
	})
}

// TestDelayedInsertTakeLonger- tests the insert behaviour with this WhenWaiting behaviour.
func TestDelayedInsertTakeIncoming(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now().Truncate(1 * time.Second))
	q := NewDelayingQueueWithCustomClock(fakeClock, "")

	first := "baz"
	second := "foo"
	third := "foo"
	fourth := "foo"
	fifth := "foo"

	q.AddAfter(first, 60*time.Second)
	// Shorter case...
	q.AddAfter(second, 5*time.Second)
	// attempt add delayed again with a shorter time time out and TakeShorter option.
	q.AddWithOptions(third, ExpandedDelayingOptions{Duration: 1 * time.Second, WhenWaiting: TakeIncoming})
	q.syncFill()
	// step past the shorter item and confirm that it got queued
	fakeClock.Step(2 * time.Second)
	if err := waitForActiveQueueDepth(q, 1); err != nil {
		t.Fatalf("expected success, got: %v", err)
	}
	assertDelayedQueueState(t, q, QueueState{
		active:  []interface{}{second},
		waiting: []interface{}{first},
	})
	item, _ := q.Get()
	q.Done(item)

	// Longer case...
	q.AddAfter(fourth, 5*time.Second)
	// attempt add delayed again with a shorter time time out and TakeIncoming option.
	q.AddWithOptions(fifth, ExpandedDelayingOptions{Duration: 20 * time.Second, WhenWaiting: TakeIncoming})
	q.syncFill()
	// step past the first waiting time... and prove the original delay got increased by the incoming.
	fakeClock.Step(10 * time.Second)
	if err := waitForActiveQueueDepth(q, 1); !wait.Interrupted(err) {
		t.Fatalf("expected timeout, got: %v", err)
	}
	// we should be left with baz in waiting and foo should now be active
	assertDelayedQueueState(t, q, QueueState{
		waiting: []interface{}{first, fifth},
	})
}

// TestRemoveWaiting
func TestRemoveWaiting(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now().Truncate(1 * time.Second))
	q := NewDelayingQueueWithCustomClock(fakeClock, "")

	first := "foo"
	second := "bar"
	third := "baz"

	q.AddAfter(first, 1*time.Second)
	q.AddAfter(second, 500*time.Millisecond)
	q.AddAfter(third, 250*time.Millisecond)
	assertDelayedQueueState(t, q, QueueState{
		waiting: []interface{}{first, second, third},
	})

	q.DoneWaiting(second)
	assertDelayedQueueState(t, q, QueueState{
		waiting: []interface{}{first, third},
	})

	// add again
	q.AddAfter(second, 500*time.Millisecond)
	assertDelayedQueueState(t, q, QueueState{
		waiting: []interface{}{first, second, third},
	})

	q.DoneWaiting(third)
	assertDelayedQueueState(t, q, QueueState{
		waiting: []interface{}{first, second},
	})

	fakeClock.Step(2 * time.Second)

	if err := waitForActiveQueueDepth(q, 2); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	actualFirst, _ := q.Get()
	q.Done(actualFirst)
	if !reflect.DeepEqual(actualFirst, second) {
		t.Errorf("expected %v, got %v", second, actualFirst)
	}
	actualSecond, _ := q.Get()
	q.Done(actualSecond)
	if !reflect.DeepEqual(actualSecond, first) {
		t.Errorf("expected %v, got %v", first, actualSecond)
	}

	assertDelayedQueueState(t, q, QueueState{})
}

func TestDeduping(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	q := NewDelayingQueueWithConfig(DelayingQueueConfig{Clock: fakeClock})

	first := "foo"

	q.AddAfter(first, 50*time.Millisecond)
	q.AddAfter(first, 70*time.Millisecond)
	q.syncFill()
	if q.Len() != 0 {
		t.Errorf("should not have added")
	}

	// step past the first block, we should receive now
	fakeClock.Step(60 * time.Millisecond)
	if err := waitForActiveQueueDepth(q, 1); err != nil {
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
	q.syncFill()
	if q.Len() != 0 {
		t.Errorf("should not have added")
	}

	fakeClock.Step(40 * time.Millisecond)
	if err := waitForActiveQueueDepth(q, 1); err != nil {
		t.Errorf("should have added")
	}
	item, _ = q.Get()
	q.Done(item)

	// step past the second add
	fakeClock.Step(20 * time.Millisecond)
	q.syncFill()
	if q.Len() != 0 {
		t.Errorf("should not have added")
	}
}

func TestAddTwoFireEarly(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	q := NewDelayingQueueWithConfig(DelayingQueueConfig{Clock: fakeClock})

	first := "foo"
	second := "bar"
	third := "baz"

	q.AddAfter(first, 1*time.Second)
	q.AddAfter(second, 50*time.Millisecond)
	q.syncFill()

	if q.Len() != 0 {
		t.Errorf("should not have added")
	}

	fakeClock.Step(60 * time.Millisecond)

	if err := waitForActiveQueueDepth(q, 1); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	item, _ := q.Get()
	if !reflect.DeepEqual(item, second) {
		t.Errorf("expected %v, got %v", second, item)
	}

	q.AddAfter(third, 2*time.Second)
	q.syncFill()

	fakeClock.Step(1 * time.Second)
	if err := waitForActiveQueueDepth(q, 1); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	item, _ = q.Get()
	if !reflect.DeepEqual(item, first) {
		t.Errorf("expected %v, got %v", first, item)
	}

	fakeClock.Step(2 * time.Second)
	if err := waitForActiveQueueDepth(q, 1); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	item, _ = q.Get()
	if !reflect.DeepEqual(item, third) {
		t.Errorf("expected %v, got %v", third, item)
	}
}

func TestCopyShifting(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	q := NewDelayingQueueWithConfig(DelayingQueueConfig{Clock: fakeClock})

	first := "foo"
	second := "bar"
	third := "baz"

	q.AddAfter(first, 1*time.Second)
	q.AddAfter(second, 500*time.Millisecond)
	q.AddAfter(third, 250*time.Millisecond)
	q.syncFill()

	if q.Len() != 0 {
		t.Errorf("should not have added")
	}

	fakeClock.Step(2 * time.Second)

	if err := waitForActiveQueueDepth(q, 3); err != nil {
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

// TestReportQueueState ensures that the queue introspection methods are working as intended.
func TestReportQueueState(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now().Truncate(1 * time.Second))
	q := NewDelayingQueueWithCustomClock(fakeClock, "")

	first := "foo"
	second := "bar"
	third := "baz"
	fourth := "apple"
	fifth := "orange"

	if q.LenWaiting() != 0 {
		t.Fatal("the length of the waiting queue must begin at 0")
	}
	waiting, itemReady := q.IsWaiting(first)
	if waiting != false {
		t.Fatal("the item foo can not be waiting on an empty queue")
	}
	if itemReady != time.Duration(0) {
		t.Fatal("the next ready time should return empty/0 when the item is not found")
	}

	// Add some items to the 'waiting' queue
	q.AddWithOptions(first, ExpandedDelayingOptions{Duration: 5 * time.Second})
	q.AddWithOptions(second, ExpandedDelayingOptions{Duration: 10 * time.Second})
	q.AddWithOptions(third, ExpandedDelayingOptions{Duration: 10 * time.Second})
	q.AddWithOptions(fourth, ExpandedDelayingOptions{Duration: 15 * time.Second})
	if q.LenWaiting() != 4 {
		t.Fatal("there should be 4 items in the waiting queue")
	}
	waiting, itemReady = q.IsWaiting(first)
	if waiting == false {
		t.Errorf("the first item foo should be reported as waiting")
	}
	if itemReady != 5*time.Second {
		t.Errorf("the first item was reported as being ready in %s where we were expecting 5 seconds", itemReady)
	}
	if waiting, _ := q.IsWaiting(fifth); waiting == true {
		t.Errorf("the fifth item should not be reported as waiting")
	}

	// stepping the clock forward 1 second
	fakeClock.Step(1 * time.Second)
	waiting, itemReady = q.IsWaiting(third)
	if waiting == false {
		t.Errorf("the third item foo should be reported as waiting")
	}
	if itemReady != 9*time.Second {
		t.Errorf("the first third item reports as being ready in %s where we were expecting 9 seconds", itemReady)
	}
}

// TestCalculatenextReadyAt tests that the nextReadyAt timer is being correctly updated when the head item on the
// 'waiting' queue changes.
func TestCalculatenextReadyAt(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now().Truncate(1 * time.Second))
	q := NewDelayingQueueWithCustomClock(fakeClock, "")

	first := "foo"
	second := "bar"
	third := "baz"
	fourth := "apple"
	fifth := "orange"

	item, ready := q.nextReady()
	if ready != 0 {
		t.Fatal("the next ready duration should start at 0 for an empty queue")
	}
	if item != nil {
		t.Fatal("there should be no next item on an empty queue")
	}

	// the next ready at should take on the value of the first item queued
	q.AddWithOptions(first, ExpandedDelayingOptions{Duration: 5 * time.Second})
	item, ready = q.nextReady()
	if item != first {
		t.Fatal("the next ready item should be the item when there is only one queued")
	}
	if ready != 5*time.Second {
		t.Fatal("the next ready time was not set correctly")
	}
	// stepping the item off the queue should update the next ready at properly
	fakeClock.Step(6 * time.Second)
	if err := waitForActiveQueueDepth(q, 1); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	_, ready = q.nextReady()
	if ready != 0 {
		t.Fatal("the next ready duration should start at 0 for an empty waiting queue")
	}

	// the next ready at should update when a new shortest item is added
	q.AddWithOptions(second, ExpandedDelayingOptions{Duration: 30 * time.Second})
	item, ready = q.nextReady()
	if item != second {
		t.Fatal("the next ready item should be the item when there is only one queued")
	}
	if ready != 30*time.Second {
		t.Fatalf("The next ready at time does not match the newly added item")
	}

	// the next ready at time should not change unless the head changes
	q.AddWithOptions(third, ExpandedDelayingOptions{Duration: 40 * time.Second})
	item, ready = q.nextReady()
	if item != second {
		t.Fatal("the second item should still be the next ready")
	}
	if ready != 30*time.Second {
		t.Fatalf("the next ready at time is not correct for the second item")
	}

	// and it should updated when the head changes...
	q.AddWithOptions(fourth, ExpandedDelayingOptions{Duration: 20 * time.Second})
	item, ready = q.nextReady()
	if item != fourth {
		t.Fatal("the forth item should be the next ready")
	}
	if ready != 20*time.Second {
		t.Fatalf("the next ready at time is not correct for the fourth item")
	}

	// step the clock forward 25 seconds
	// second now has 5 seconds left
	// third now has 15 seconds left
	// fourth is now ready and pops to 'active' queue
	fakeClock.Step(25 * time.Second)
	if err := waitForActiveQueueDepth(q, 2); err != nil { // active queue should contain foo and baz
		t.Fatalf("unexpected err: %v", err)
	}
	item, ready = q.nextReady()
	if item != second {
		t.Fatal("the second item should be the next ready")
	}
	if ready != 5*time.Second {
		t.Fatalf("the next ready at time is not correct for the second item")
	}

	// forgetting/deleting the head item should update the next ready at time.
	q.DoneWaiting(second)
	item, ready = q.nextReady()
	if item != third {
		t.Fatal("the third item should be the next ready")
	}
	if ready != 15*time.Second {
		t.Fatalf("the next ready at time is not correct for the third item")
	}

	// the next ready at time should change when an immediate item pre-empts the 'waiting' item
	q.AddWithOptions(fifth, ExpandedDelayingOptions{Duration: 30 * time.Second})
	q.AddWithOptions(third, ExpandedDelayingOptions{})
	item, ready = q.nextReady()
	if item != fifth {
		t.Fatal("the fifth item should be the next ready")
	}
	if ready != 30*time.Second {
		t.Fatalf("the next ready at time is not correct for the fifth item")
	}
}

func BenchmarkDelayingQueue_AddAfter(b *testing.B) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	q := NewDelayingQueueWithConfig(DelayingQueueConfig{Clock: fakeClock})

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

// waitForActiveQueueDepth will poll until the active queue depth matches the desired depth, or times out.
func waitForActiveQueueDepth(q DelayingInterface, depth int) error {
	ctx := context.TODO()
	return wait.PollUntilContextTimeout(ctx, DefaultWaitStep, DefaultWaitTimeout, true, func(ctx context.Context) (done bool, err error) {
		if q.Len() == depth {
			return true, nil
		}

		return false, nil
	})
}

// QueueState provides a convenient way to confirm the both the Active and Waiting parts of the delaying queue
// are in the correct state
type QueueState struct {
	active  []interface{}
	waiting []interface{}
}

// assertQueueState allows us to assert exactly what items we expect to be in the 'active' and 'waiting' queues.
// This is more declarative and thorough than observing all of the behaviours from the 'active' queue alone.
func assertDelayedQueueState(t *testing.T, q *delayingType, qs QueueState) {
	var errored bool

	// check the 'waiting' queue
	wlen := q.LenWaiting()
	if wlen != len(qs.waiting) {
		t.Errorf("the waiting queue has %d queued items but we expected %d", wlen, len(qs.waiting))
		errored = true
	}
	for _, w := range qs.waiting {
		waiting, _ := q.IsWaiting(w)
		if !waiting {
			t.Errorf("item %v was expected to be waiting but isn't", w)
			errored = true
		}
	}

	// check 'active' queue
	if q.Len() != len(qs.active) {
		t.Errorf("the active q has %d queued items but we expected %d", q.Len(), len(qs.active))
		errored = true
	}
	for _, a := range qs.active {
		if !q.IsQueued(a) {
			t.Errorf("item %v was expected to be actively queued but isn't", a)
			errored = true
		}
	}

	if errored {
		t.Fatalf("assertQueueState (%+v) does not match actual", qs)
	}
}
