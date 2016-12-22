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
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/util/workqueue"
)

func TestBasic(t *testing.T) {
	// If something is seriously wrong this test will never complete.
	q := workqueue.New()

	// Start producers
	const producers = 50
	producerWG := sync.WaitGroup{}
	producerWG.Add(producers)
	for i := 0; i < producers; i++ {
		go func(i int) {
			defer producerWG.Done()
			for j := 0; j < 50; j++ {
				q.Add(i)
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
				item, quit := q.Get()
				if item == "added after shutdown!" {
					t.Errorf("Got an item added after shutdown.")
				}
				if quit {
					return
				}
				t.Logf("Worker %v: begin processing %v", i, item)
				time.Sleep(3 * time.Millisecond)
				t.Logf("Worker %v: done processing %v", i, item)
				q.Done(item)
			}
		}(i)
	}

	producerWG.Wait()
	q.ShutDown()
	q.Add("added after shutdown!")
	consumerWG.Wait()
}

func TestAddWhileProcessing(t *testing.T) {
	q := workqueue.New()

	// Start producers
	const producers = 50
	producerWG := sync.WaitGroup{}
	producerWG.Add(producers)
	for i := 0; i < producers; i++ {
		go func(i int) {
			defer producerWG.Done()
			q.Add(i)
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
				item, quit := q.Get()
				if quit {
					return
				}
				counters[item]++
				if counters[item] < 2 {
					q.Add(item)
				}
				q.Done(item)
			}
		}(i)
	}

	producerWG.Wait()
	q.ShutDown()
	consumerWG.Wait()
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
