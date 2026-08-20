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

package workqueue_test

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
)

// traceQueue traces whether items are touched
type traceQueue struct {
	workqueue.Queue[any]

	touched map[interface{}]struct{}
}

func (t *traceQueue) Touch(item interface{}) {
	t.Queue.Touch(item)
	if t.touched == nil {
		t.touched = make(map[interface{}]struct{})
	}
	t.touched[item] = struct{}{}
}

var _ workqueue.Queue[any] = &traceQueue{}

func TestBasic(t *testing.T) {
	tests := []struct {
		queue         *workqueue.Type
		queueShutDown func(workqueue.Interface)
	}{
		{
			queue:         workqueue.New(),
			queueShutDown: workqueue.Interface.ShutDown,
		},
		{
			queue:         workqueue.New(),
			queueShutDown: workqueue.Interface.ShutDownWithDrain,
		},
	}
	for _, test := range tests {
		// If something is seriously wrong this test will never complete.

		// Start producers
		const producers = 50
		producerWG := sync.WaitGroup{}
		producerWG.Add(producers)
		for i := 0; i < producers; i++ {
			go func(i int) {
				defer producerWG.Done()
				for j := 0; j < 50; j++ {
					test.queue.Add(i)
					time.Sleep(time.Millisecond)
				}
			}(i)
		}

		// Start consumers
		const consumers = 10
		consumerWG := sync.WaitGroup{}
		consumerWG.Add(consumers)
		for i := 0; i < consumers; i++ {
			go func(i int) {
				defer consumerWG.Done()
				for {
					item, quit := test.queue.Get()
					if item == "added after shutdown!" {
						t.Errorf("Got an item added after shutdown.")
					}
					if quit {
						return
					}
					t.Logf("Worker %v: begin processing %v", i, item)
					time.Sleep(3 * time.Millisecond)
					t.Logf("Worker %v: done processing %v", i, item)
					test.queue.Done(item)
				}
			}(i)
		}

		producerWG.Wait()
		test.queueShutDown(test.queue)
		test.queue.Add("added after shutdown!")
		consumerWG.Wait()
		if test.queue.Len() != 0 {
			t.Errorf("Expected the queue to be empty, had: %v items", test.queue.Len())
		}
	}
}

func TestAddWhileProcessing(t *testing.T) {
	tests := []struct {
		queue         *workqueue.Type
		queueShutDown func(workqueue.Interface)
	}{
		{
			queue:         workqueue.New(),
			queueShutDown: workqueue.Interface.ShutDown,
		},
		{
			queue:         workqueue.New(),
			queueShutDown: workqueue.Interface.ShutDownWithDrain,
		},
	}
	for _, test := range tests {

		// Start producers
		const producers = 50
		producerWG := sync.WaitGroup{}
		producerWG.Add(producers)
		for i := 0; i < producers; i++ {
			go func(i int) {
				defer producerWG.Done()
				test.queue.Add(i)
			}(i)
		}

		// Start consumers
		const consumers = 10
		consumerWG := sync.WaitGroup{}
		consumerWG.Add(consumers)
		for i := 0; i < consumers; i++ {
			go func(i int) {
				defer consumerWG.Done()
				// Every worker will re-add every item up to two times.
				// This tests the dirty-while-processing case.
				counters := map[interface{}]int{}
				for {
					item, quit := test.queue.Get()
					if quit {
						return
					}
					counters[item]++
					if counters[item] < 2 {
						test.queue.Add(item)
					}
					test.queue.Done(item)
				}
			}(i)
		}

		producerWG.Wait()
		test.queueShutDown(test.queue)
		consumerWG.Wait()
		if test.queue.Len() != 0 {
			t.Errorf("Expected the queue to be empty, had: %v items", test.queue.Len())
		}
	}
}

func TestLen(t *testing.T) {
	q := workqueue.New()
	q.Add("foo")
	if e, a := 1, q.Len(); e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	q.Add("bar")
	if e, a := 2, q.Len(); e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	q.Add("foo") // should not increase the queue length.
	if e, a := 2, q.Len(); e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
}

func TestReinsert(t *testing.T) {
	q := workqueue.New()
	q.Add("foo")

	// Start processing
	i, _ := q.Get()
	if i != "foo" {
		t.Errorf("Expected %v, got %v", "foo", i)
	}

	// Add it back while processing
	q.Add(i)

	// Finish it up
	q.Done(i)

	// It should be back on the queue
	i, _ = q.Get()
	if i != "foo" {
		t.Errorf("Expected %v, got %v", "foo", i)
	}

	// Finish that one up
	q.Done(i)

	if a := q.Len(); a != 0 {
		t.Errorf("Expected queue to be empty. Has %v items", a)
	}
}

func TestCollapse(t *testing.T) {
	tq := &traceQueue{Queue: workqueue.DefaultQueue[any]()}
	q := workqueue.NewWithConfig(workqueue.QueueConfig{
		Name:  "",
		Queue: tq,
	})
	// Add a new one twice
	q.Add("bar")
	q.Add("bar")

	// It should get the new one
	i, _ := q.Get()
	if i != "bar" {
		t.Errorf("Expected %v, got %v", "bar", i)
	}

	// Finish that one up
	q.Done(i)

	// There should be no more objects in the queue
	if a := q.Len(); a != 0 {
		t.Errorf("Expected queue to be empty. Has %v items", a)
	}

	if _, ok := tq.touched["bar"]; !ok {
		t.Errorf("Expected bar to be Touched")
	}
}

func TestCollapseWhileProcessing(t *testing.T) {
	tq := &traceQueue{Queue: workqueue.DefaultQueue[any]()}
	q := workqueue.NewWithConfig(workqueue.QueueConfig{
		Name:  "",
		Queue: tq,
	})
	q.Add("foo")

	// Start processing
	i, _ := q.Get()
	if i != "foo" {
		t.Errorf("Expected %v, got %v", "foo", i)
	}

	// Add the same one twice
	q.Add("foo")
	q.Add("foo")

	waitCh := make(chan struct{})
	// simulate another worker consuming the queue
	go func() {
		defer close(waitCh)
		i, _ := q.Get()
		if i != "foo" {
			t.Errorf("Expected %v, got %v", "foo", i)
		}
		// Finish that one up
		q.Done(i)
	}()

	// give the worker some head start to avoid races
	// on the select statement that cause flakiness
	time.Sleep(100 * time.Millisecond)
	// Finish the first one to unblock the other worker
	select {
	case <-waitCh:
		t.Errorf("worker should be blocked until we are done")
	default:
		q.Done("foo")
	}

	// wait for the worker to consume the new object
	// There should be no more objects in the queue
	<-waitCh
	if a := q.Len(); a != 0 {
		t.Errorf("Expected queue to be empty. Has %v items", a)
	}

	if _, ok := tq.touched["foo"]; ok {
		t.Errorf("Unexpected Touch")
	}
}

func TestQueueDrainageUsingShutDownWithDrain(t *testing.T) {

	q := workqueue.New()

	q.Add("foo")
	q.Add("bar")

	firstItem, _ := q.Get()
	secondItem, _ := q.Get()

	finishedWG := sync.WaitGroup{}
	finishedWG.Add(1)
	go func() {
		defer finishedWG.Done()
		q.ShutDownWithDrain()
	}()

	// This is done as to simulate a sequence of events where ShutDownWithDrain
	// is called before we start marking all items as done - thus simulating a
	// drain where we wait for all items to finish processing.
	shuttingDown := false
	for !shuttingDown {
		_, shuttingDown = q.Get()
	}

	// Mark the first two items as done, as to finish up
	q.Done(firstItem)
	q.Done(secondItem)

	finishedWG.Wait()
}

func TestNoQueueDrainageUsingShutDown(t *testing.T) {

	q := workqueue.New()

	q.Add("foo")
	q.Add("bar")

	q.Get()
	q.Get()

	finishedWG := sync.WaitGroup{}
	finishedWG.Add(1)
	go func() {
		defer finishedWG.Done()
		// Invoke ShutDown: suspending the execution immediately.
		q.ShutDown()
	}()

	// We can now do this and not have the test timeout because we didn't call
	// Done on the first two items before arriving here.
	finishedWG.Wait()
}

func TestForceQueueShutdownUsingShutDown(t *testing.T) {

	q := workqueue.New()

	q.Add("foo")
	q.Add("bar")

	q.Get()
	q.Get()

	finishedWG := sync.WaitGroup{}
	finishedWG.Add(1)
	go func() {
		defer finishedWG.Done()
		q.ShutDownWithDrain()
	}()

	// This is done as to simulate a sequence of events where ShutDownWithDrain
	// is called before ShutDown
	shuttingDown := false
	for !shuttingDown {
		_, shuttingDown = q.Get()
	}

	// Use ShutDown to force the queue to shut down (simulating a caller
	// which can invoke this function on a second SIGTERM/SIGINT)
	q.ShutDown()

	// We can now do this and not have the test timeout because we didn't call
	// done on any of the items before arriving here.
	finishedWG.Wait()
}

func TestQueueDrainageUsingShutDownWithDrainWithDirtyItem(t *testing.T) {
	q := workqueue.New()

	q.Add("foo")
	gotten, _ := q.Get()
	q.Add("foo")

	finishedWG := sync.WaitGroup{}
	finishedWG.Add(1)
	go func() {
		defer finishedWG.Done()
		q.ShutDownWithDrain()
	}()

	// Ensure that ShutDownWithDrain has started and is blocked.
	shuttingDown := false
	for !shuttingDown {
		_, shuttingDown = q.Get()
	}

	// Finish "working".
	q.Done(gotten)

	// `shuttingDown` becomes false because Done caused an item to go back into
	// the queue.
	again, shuttingDown := q.Get()
	if shuttingDown {
		t.Fatalf("should not have been done")
	}
	q.Done(again)

	// Now we are really done.
	_, shuttingDown = q.Get()
	if !shuttingDown {
		t.Fatalf("should have been done")
	}

	finishedWG.Wait()
}

// TestGarbageCollection ensures that objects that are added then removed from the queue are
// able to be garbage collected.
func TestGarbageCollection(t *testing.T) {
	type bigObject struct {
		data []byte
	}
	leakQueue := workqueue.New()
	t.Cleanup(func() {
		// Make sure leakQueue doesn't go out of scope too early
		runtime.KeepAlive(leakQueue)
	})
	c := &bigObject{data: []byte("hello")}
	mustGarbageCollect(t, c)
	leakQueue.Add(c)
	o, _ := leakQueue.Get()
	leakQueue.Done(o)
}

// mustGarbageCollect asserts than an object was garbage collected by the end of the test.
// The input must be a pointer to an object.
func mustGarbageCollect(t *testing.T, i interface{}) {
	t.Helper()
	var collected int32 = 0
	runtime.SetFinalizer(i, func(x interface{}) {
		atomic.StoreInt32(&collected, 1)
	})
	t.Cleanup(func() {
		if err := wait.PollImmediate(time.Millisecond*100, wait.ForeverTestTimeout, func() (done bool, err error) {
			// Trigger GC explicitly, otherwise we may need to wait a long time for it to run
			runtime.GC()
			return atomic.LoadInt32(&collected) == 1, nil
		}); err != nil {
			t.Errorf("object was not garbage collected")
		}
	})
}

// TestShutDownWithDrainWaitsForQueuedItems checks that ShutDownWithDrain does not
// return while items remain in the queue (not only while items are in processing).
func TestShutDownWithDrainWaitsForQueuedItems(t *testing.T) {
	q := workqueue.New()
	q.Add("a")
	q.Add("b")

	first, _ := q.Get() // processing={a}, queue=[b]

	// Worker takes remaining items but holds "b" in processing until released so
	// we can assert drain stays blocked without relying on timing.
	holdB := make(chan struct{})
	sawB := make(chan struct{})
	workerDone := make(chan struct{})
	go func() {
		defer close(workerDone)
		for {
			item, quit := q.Get()
			if quit {
				return
			}
			if item == "b" {
				close(sawB)
				<-holdB
			}
			q.Done(item)
		}
	}()

	drained := make(chan struct{})
	go func() {
		q.ShutDownWithDrain()
		close(drained)
	}()

	// Finish "a". Worker should receive queued "b" and block before Done.
	q.Done(first)

	select {
	case <-sawB:
	case <-drained:
		t.Fatalf("ShutDownWithDrain returned before remaining item was processed (len=%d)", q.Len())
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("timed out waiting for worker to get queued item b")
	}

	// "b" is in-flight; drain must still be waiting.
	select {
	case <-drained:
		t.Fatal("ShutDownWithDrain returned while item b still processing")
	default:
	}

	close(holdB)

	select {
	case <-drained:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("ShutDownWithDrain did not return after queue drained")
	}

	select {
	case <-workerDone:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("worker did not exit after drain")
	}

	if q.Len() != 0 {
		t.Fatalf("expected empty queue after drain, len=%d", q.Len())
	}
}

// TestShutDownWithDrainManyWorkers stresses ShutDownWithDrain with multiple workers
// so a single Signal cannot leave drain or workers stuck.
func TestShutDownWithDrainManyWorkers(t *testing.T) {
	q := workqueue.New()
	for i := 0; i < 50; i++ {
		q.Add(i)
	}

	const workers = 8
	var started, wg sync.WaitGroup
	started.Add(workers)
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func() {
			defer wg.Done()
			started.Done()
			for {
				item, quit := q.Get()
				if quit {
					return
				}
				q.Done(item)
			}
		}()
	}
	started.Wait()

	drained := make(chan struct{})
	go func() {
		q.ShutDownWithDrain()
		close(drained)
	}()

	select {
	case <-drained:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("deadlock: ShutDownWithDrain did not return")
	}

	workersDone := make(chan struct{})
	go func() {
		wg.Wait()
		close(workersDone)
	}()
	select {
	case <-workersDone:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("deadlock: workers did not exit after drain")
	}
}

// TestDoneIdempotent ensures a spurious extra Done does not duplicate queue entries.
func TestDoneIdempotent(t *testing.T) {
	q := workqueue.New()
	q.Add("x")
	item, _ := q.Get()
	q.Add(item) // dirty while processing
	q.Done(item)
	q.Done(item) // spurious second Done

	if got := q.Len(); got != 1 {
		t.Fatalf("queue len=%d after requeue + spurious Done, want 1", got)
	}

	got, quit := q.Get()
	if quit || got != "x" {
		t.Fatalf("Get()=(%v, quit=%v), want (x, false)", got, quit)
	}
	if q.Len() != 0 {
		t.Fatalf("queue len=%d after one Get, want 0 (duplicate entries)", q.Len())
	}
	q.Done(got)

	q.ShutDown()
	if _, quit = q.Get(); !quit {
		t.Fatal("expected shutdown signal with empty queue")
	}
}

func BenchmarkQueue(b *testing.B) {
	keys := make([]string, 100)
	for idx := range keys {
		keys[idx] = fmt.Sprintf("key-%d", idx)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		q := workqueue.NewTypedWithConfig(workqueue.TypedQueueConfig[string]{})
		b.StartTimer()
		for j := 0; j < 100; j++ {
			q.Add(keys[j])
			key, _ := q.Get()
			q.Done(key)
		}
	}
}
