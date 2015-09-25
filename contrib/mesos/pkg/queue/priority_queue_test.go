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

package queue

import (
	"testing"
	"time"
)

func timedPriority(t time.Time) Priority {
	return Priority{ts: t}
}

func TestPQ(t *testing.T) {
	t.Parallel()

	var pq priorityQueue
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}

	now := timedPriority(time.Now())
	now2 := timedPriority(now.ts.Add(2 * time.Second))
	pq.Push(&qitem{priority: now2})
	if pq.Len() != 1 {
		t.Fatalf("pq.len should be 1")
	}
	x := pq.Pop()
	if x == nil {
		t.Fatalf("x is nil")
	}
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}
	item := x.(*qitem)
	if !item.priority.Equal(now2) {
		t.Fatalf("item.priority != now2")
	}

	pq.Push(&qitem{priority: now2})
	pq.Push(&qitem{priority: now2})
	pq.Push(&qitem{priority: now2})
	pq.Push(&qitem{priority: now2})
	pq.Push(&qitem{priority: now2})
	pq.Pop()
	pq.Pop()
	pq.Pop()
	pq.Pop()
	pq.Pop()
	if pq.Len() != 0 {
		t.Fatalf("pq should be empty")
	}
	now4 := timedPriority(now.ts.Add(4 * time.Second))
	now6 := timedPriority(now.ts.Add(4 * time.Second))
	pq.Push(&qitem{priority: now2})
	pq.Push(&qitem{priority: now4})
	pq.Push(&qitem{priority: now6})
	pq.Swap(0, 2)
	if !pq[0].priority.Equal(now6) || !pq[2].priority.Equal(now2) {
		t.Fatalf("swap failed")
	}
	if pq.Less(1, 2) {
		t.Fatalf("now4 < now2")
	}
}

func TestPopEmptyPQ(t *testing.T) {
	t.Parallel()
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("Expected panic from popping an empty PQ")
		}
	}()
	var pq priorityQueue
	pq.Pop()
}
