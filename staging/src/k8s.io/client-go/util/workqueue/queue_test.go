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

	"github.com/onsi/gomega"

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
		// The test context is canceled before cleanup callbacks run.
		gomega.NewWithT(t).Eventually(func() int32 {
			// Trigger GC explicitly, otherwise we may need to wait a long time for it to run
			runtime.GC()
			return atomic.LoadInt32(&collected)
		}).WithTimeout(wait.ForeverTestTimeout).WithPolling(100*time.Millisecond).Should(gomega.Equal(int32(1)), "object was not garbage collected")
	})
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

func TestQueuePauseAndResume(t *testing.T) {
	t.Run("Get blocks while paused and unblocks on Resume, Add and Done work while paused", func(t *testing.T) {
		q := workqueue.NewTyped[string]()
		defer q.ShutDown()

		// Dequeue an item before pausing so we can call Done while paused.
		q.Add("in-flight")
		inFlight, shutdown := q.Get()
		if shutdown || inFlight != "in-flight" {
			t.Fatalf("expected in-flight, got %q (shutdown=%v)", inFlight, shutdown)
		}

		q.Pause()

		// Re-add in-flight item and add a new item while paused.
		q.Add("in-flight")
		q.Done(inFlight)
		q.Add("queued-while-paused")

		if q.Len() != 2 {
			t.Fatalf("expected queue length 2 while paused, got %d", q.Len())
		}

		gotCh := make(chan string, 2)
		go func() {
			item1, quit := q.Get()
			if quit {
				return
			}
			q.Done(item1)
			gotCh <- item1

			item2, quit := q.Get()
			if quit {
				return
			}
			q.Done(item2)
			gotCh <- item2
		}()

		select {
		case item := <-gotCh:
			t.Fatalf("Get returned %q while queue was paused", item)
		case <-time.After(50 * time.Millisecond):
		}

		q.Resume()

		for _, want := range []string{"in-flight", "queued-while-paused"} {
			select {
			case got := <-gotCh:
				if got != want {
					t.Fatalf("expected %q, got %q", want, got)
				}
			case <-time.After(time.Second):
				t.Fatalf("timed out waiting for %q after Resume", want)
			}
		}
	})

	t.Run("ShutDown unblocks a paused Get", func(t *testing.T) {
		q := workqueue.NewTyped[string]()
		q.Add("item-1")
		q.Pause()

		shutdownCh := make(chan bool, 1)
		go func() {
			_, shutdown := q.Get()
			shutdownCh <- shutdown
		}()

		select {
		case <-shutdownCh:
			t.Fatalf("Get returned before ShutDown while paused")
		case <-time.After(50 * time.Millisecond):
		}

		q.ShutDown()

		select {
		case shutdown := <-shutdownCh:
			if !shutdown {
				t.Fatalf("expected shutdown=true from paused Get after ShutDown")
			}
		case <-time.After(time.Second):
			t.Fatalf("timed out waiting for paused Get to unblock on ShutDown")
		}
	})

	t.Run("ShutDownWithDrain on a paused queue waits for in-flight items and does not hand out queued ones", func(t *testing.T) {
		q := workqueue.NewTyped[string]()
		q.Add("in-flight")
		inFlight, shutdown := q.Get()
		if shutdown || inFlight != "in-flight" {
			t.Fatalf("expected in-flight, got %q (shutdown=%v)", inFlight, shutdown)
		}
		q.Pause()
		q.Add("queued-while-paused")

		workerCh := make(chan bool, 1)
		go func() {
			_, shutdown := q.Get()
			workerCh <- shutdown
		}()

		drained := make(chan struct{})
		go func() {
			q.ShutDownWithDrain()
			close(drained)
		}()

		// The worker is told to exit without receiving the queued item.
		select {
		case shutdown := <-workerCh:
			if !shutdown {
				t.Fatalf("expected shutdown=true from paused Get after ShutDownWithDrain")
			}
		case <-time.After(time.Second):
			t.Fatalf("timed out waiting for paused Get to unblock on ShutDownWithDrain")
		}

		// Drain waits for the in-flight item to be marked Done.
		select {
		case <-drained:
			t.Fatalf("ShutDownWithDrain returned while an item was still in flight")
		case <-time.After(50 * time.Millisecond):
		}
		q.Done(inFlight)
		select {
		case <-drained:
		case <-time.After(time.Second):
			t.Fatalf("timed out waiting for ShutDownWithDrain to return after Done")
		}
	})

	t.Run("rate limiting and delaying queues forward Pause and Resume", func(t *testing.T) {
		q := workqueue.NewTypedRateLimitingQueue[string](workqueue.DefaultTypedControllerRateLimiter[string]())
		defer q.ShutDown()
		p, ok := workqueue.TypedInterface[string](q).(workqueue.Pausable)
		if !ok {
			t.Fatalf("expected rate limiting queue to implement Pausable")
		}

		q.Add("item")
		p.Pause()

		gotCh := make(chan string, 1)
		go func() {
			item, quit := q.Get()
			if !quit {
				q.Done(item)
				gotCh <- item
			}
		}()
		select {
		case item := <-gotCh:
			t.Fatalf("Get returned %q while the rate limiting queue was paused", item)
		case <-time.After(50 * time.Millisecond):
		}

		p.Resume()
		select {
		case got := <-gotCh:
			if got != "item" {
				t.Fatalf("expected item, got %q", got)
			}
		case <-time.After(time.Second):
			t.Fatalf("timed out waiting for Get after Resume")
		}
	})
}
