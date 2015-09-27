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

package delay

import (
	"testing"
	"time"

	"github.com/pivotal-golang/clock/fakeclock"

	"k8s.io/kubernetes/contrib/mesos/pkg/queue/priority"
)

func TestPQWithDelayPriority(t *testing.T) {
	t.Parallel()

	pq := priority.NewPriorityQueue()
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}

	clock := fakeclock.NewFakeClock(time.Now())
	start := clock.Now()

	now2 := NewPriority(start.Add(2 * time.Second))
	pq.Push(priority.NewItem(nil, now2))
	if pq.Len() != 1 {
		t.Fatalf("pq.len should be 1")
	}

	popCh := make(chan interface{})
	go func() {
		// blocks until 2 seconds have passed
		popCh <- pq.Pop()
	}()
	clock.Increment(2 * time.Second)
	x := <-popCh
	if x == nil {
		t.Fatalf("x is nil")
	}
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}
	item := x.(priority.Item)
	if !item.Priority().Equal(now2) {
		t.Fatalf("item.Priority != now2")
	}

	pq.Push(priority.NewItem(nil, now2))
	pq.Push(priority.NewItem(nil, now2))
	pq.Push(priority.NewItem(nil, now2))
	pq.Push(priority.NewItem(nil, now2))
	pq.Push(priority.NewItem(nil, now2))
	if pq.Len() != 5 {
		t.Fatalf("pq.len should be 5")
	}

	// none of these pops block, because the event time has already elapsed
	pq.Pop()
	pq.Pop()
	pq.Pop()
	pq.Pop()
	pq.Pop()
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}

	now4 := NewPriority(start.Add(4 * time.Second))
	now6 := NewPriority(start.Add(6 * time.Second))
	pq.Push(priority.NewItem(nil, now2))
	pq.Push(priority.NewItem(nil, now4))
	pq.Push(priority.NewItem(nil, now6))

	qList := *pq
	// list order based on insertion (not time)
	if !qList[0].Priority().Equal(now2) {
		t.Fatalf("now2 should be first")
	}
	if !qList[1].Priority().Equal(now4) {
		t.Fatalf("now4 should be second")
	}
	if !qList[2].Priority().Equal(now6) {
		t.Fatalf("now4 should be third")
	}

	pq.Swap(0, 2)
	if !qList[0].Priority().Equal(now6) || !qList[2].Priority().Equal(now2) {
		t.Fatalf("swap failed")
	}
	if pq.Less(1, 2) {
		t.Fatalf("now4 < now2")
	}
}
